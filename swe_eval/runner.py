import hashlib
from pathlib import Path
import re
import time

from .config import ROOT, digest
from .doctor import doctor
from .io import execute, read_json, redact, scrub_tree, write_json
from .results import row, summarize, swe_result, tb_results
from .metrics import docker_oom_events, enrich, task_directory


def environment_python(benchmark):
    project = ROOT / 'environments' / benchmark
    if not (project / 'uv.lock').exists() or not (project / '.venv' / 'bin' / 'python').exists():
        raise RuntimeError(f'{benchmark} environment is not locked/installed; run bash scripts/setup.sh lock, then bash scripts/setup.sh sync')
    return str(project / '.venv' / 'bin' / 'python')


def worker(benchmark, args, config, log, cwd=ROOT, timeout=300):
    result = execute([environment_python(benchmark), '-m', 'swe_eval.worker', *map(str, args)],
                     cwd, log, env=config.child_env(), timeout=timeout)
    if result['timed_out'] or result['returncode']:
        raise RuntimeError(f'Worker {args[0]} failed; see {log}')
    return result


def validate_manifest(manifest, benchmark, profile):
    n = 1 if profile == 'smoke' else 10
    tasks = manifest['tasks']
    ids = [t['id'] for t in tasks]
    if manifest['benchmark'] != benchmark or manifest['profile'] != profile or len(ids) != n:
        raise ValueError('Manifest benchmark, profile or task count mismatch')
    if ids != sorted(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', x) or x in ('.', '..') for x in ids):
        raise ValueError('Manifest IDs must be unique, sorted and safe path components')
    if manifest['tasks_sha256'] != digest(tasks):
        raise ValueError('Manifest content hash mismatch')
    return manifest


def prepare(benchmark, profile, config):
    path = ROOT / 'profiles' / f'{benchmark}-{profile}.json'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use an exclusive marker to prevent competing preparation overwrites.
        marker = path.with_suffix('.preparing')
        try:
            with marker.open('x'):
                temporary = path.with_suffix('.candidate.json')
                try:
                    worker(benchmark, ['prepare', benchmark, profile, temporary], config,
                           path.with_suffix('.prepare.log'), timeout=900)
                    validate_manifest(read_json(temporary), benchmark, profile)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        finally:
            # Only remove our own marker (FileExistsError doesn't authorize removing another's).
            if 'temporary' in locals():
                marker.unlink(missing_ok=True)
    return validate_manifest(read_json(path), benchmark, profile)


def tb_config(config, tasks, output, oracle):
    agent = {'name': 'oracle'} if oracle else {
        'name': 'terminus-2', 'model_name': 'openai/' + config.model,
        'kwargs': {'api_base': config.api_base, 'max_turns': config.max_turns, 'parser_name': 'json',
                   'temperature': config.temperature, 'interleaved_thinking': False,
                   'store_all_messages': True, 'enable_summarize': False,
                   'record_terminal_session': config.record_terminal_session,
                   'model_info': {'max_input_tokens': config.context_length,
                                  'max_output_tokens': config.max_tokens,
                                  'input_cost_per_token': 0, 'output_cost_per_token': 0},
                   'llm_call_kwargs': dict(timeout=config.api_timeout, num_retries=0, **config.sampling())}}
    return {'job_name': 'job', 'jobs_dir': str(output / 'harbor'), 'n_attempts': 1,
            'n_concurrent_trials': 1, 'retry': {'max_retries': 0},
            'environment': {'type': 'docker', 'delete': True}, 'agents': [agent],
            'tasks': [task['task_config'] for task in tasks]}


def run_swe(config, tasks, output, oracle):
    rows = []
    if not oracle:
        worker('swebench', ['swe-config', output / 'config.json', output / 'agent-config.json'],
               config, output / 'config.log')
    for task in tasks:
        task_id = task['id']
        print(f'SWE-bench: {task_id}', flush=True)
        directory = output / 'tasks' / task_id
        directory.mkdir(parents=True)
        write_json(directory / 'task.json', task['record'])
        write_json(directory / 'dataset.json', [task['record']])
        agent_dir = directory / 'agent'
        agent_dir.mkdir()
        scoring = directory / 'scoring'
        scoring.mkdir()
        started = time.monotonic()
        if oracle:
            write_json(agent_dir / 'preds.json', {task_id: {'instance_id': task_id,
                       'model_name_or_path': 'gold', 'model_patch': task['record']['patch']}})
            process = {'returncode': 0, 'timed_out': False, 'elapsed_seconds': 0}
        else:
            process = execute([environment_python('swebench'), '-m', 'swe_eval.worker', 'swe-agent',
                               str(output / 'agent-config.json'), str(directory / 'task.json'), str(agent_dir)],
                              ROOT, agent_dir / 'agent.log', config.child_env(), config.swe_agent_timeout)
        write_json(directory / 'agent-process.json', process)
        before = swe_result(task_id, agent_dir, scoring, process, oracle)
        # No patch can be graded if agent failed to submit. Preserve its terminal status.
        budget_submission = (before['status'] == 'budget_exhausted' and not process['timed_out']
                             and (agent_dir / 'preds.json').exists())
        if not oracle and before['detail'] != 'Missing or duplicate official harness report' and not budget_submission:
            rows.append(before)
            write_json(output / 'results.json', rows)
            continue
        predictions = read_json(agent_dir / 'preds.json')
        # Safe, stable harness label. Actual model/checkpoint is recorded in config.json.
        predictions[task_id]['model_name_or_path'] = 'gold' if oracle else 'candidate'
        write_json(directory / 'predictions.json', predictions)
        score_process = execute([environment_python('swebench'), '-m', 'swebench.harness.run_evaluation',
                                 '--dataset_name', str(directory / 'dataset.json'), '--split', 'test',
                                 '--predictions_path', str(directory / 'predictions.json'),
                                 '--instance_ids', task_id, '--max_workers', '1',
                                 '--run_id', 'score', '--timeout', str(config.swe_score_timeout)],
                                scoring, scoring / 'harness.log', config.child_env(), config.swe_score_timeout)
        write_json(directory / 'score-process.json', score_process)
        result = swe_result(task_id, agent_dir, scoring, process, oracle)
        if score_process['returncode'] or score_process['timed_out']:
            result.update(status='environment_error', resolved=None,
                          detail='Official harness timed out; see harness.log' if score_process['timed_out']
                          else 'Official harness process failed; see harness.log')
        elif budget_submission and result['resolved'] is None:
            result.update(status='environment_error', detail='Missing official harness outcome for exhausted budget')
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        rows.append(result)
        write_json(output / 'results.json', rows)
    return rows


def run(benchmark, profile, run_id, config, runs_dir=Path('runs'), oracle=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('run-id must be a safe directory name')
    output = (runs_dir / run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    started_at = time.time()
    expected = 1 if profile == 'smoke' else 10
    rows, fatal = [], None
    write_json(output / 'config.json', {**config.snapshot()})
    write_json(output / 'run.json', {'benchmark': benchmark, 'profile': profile, 'run_id': run_id,
                                   'oracle': oracle, 'started_at_unix': time.time()})
    print(f'Artifacts: {output}', flush=True)
    try:
        manifest = prepare(benchmark, profile, config)
        tasks = manifest['tasks']
        write_json(output / 'manifest.json', manifest)
        rows = [row(t['id']) for t in tasks]
        write_json(output / 'results.json', rows)
        worker(benchmark, ['metadata', benchmark, output / 'dependencies.json'], config, output / 'metadata.log')
        lock = ROOT / 'environments' / benchmark / 'uv.lock'
        (output / 'environment.uv.lock').write_bytes(lock.read_bytes())
        metadata = read_json(output / 'dependencies.json')
        frozen = config.snapshot()
        for key in ('model', 'model_path', 'checkpoint', 'api_base', 'model_instance', 'api_key_env'):
            frozen.pop(key)
        write_json(output / 'comparison.json', {
            'benchmark': benchmark, 'tasks_sha256': manifest['tasks_sha256'],
            'settings': frozen, 'oracle': oracle, 'concurrency': 1, 'attempts': 1,
            'versions': metadata['versions'], 'agent_sources_sha256': metadata['agent_sources_sha256'],
            'installed_distributions_sha256': digest(metadata['all_distributions']),
            'python': metadata['python'],
            'environment_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest(),
            'runner_sha256': digest({p.name: p.read_text() for p in sorted((ROOT / 'swe_eval').glob('*.py'))})})
        preflight = doctor(config, api=not oracle)
        write_json(output / 'doctor.json', preflight)
        if not preflight['ok']:
            for result in rows:
                result.update(status='environment_error' if any(not c['ok'] for k, c in preflight['checks'].items() if k != 'api')
                              else 'api_error', detail='Preflight failed; see doctor.json')
            raise RuntimeError('Preflight failed; see doctor.json')
        if benchmark == 'swebench':
            rows = run_swe(config, tasks, output, oracle)
        else:
            write_json(output / 'harbor-config.json', tb_config(config, tasks, output, oracle))
            process = execute([environment_python(benchmark), '-m', 'swe_eval.worker', 'tb-run',
                               str(output / 'harbor-config.json')], output, output / 'harbor.log',
                              config.child_env(), timeout=None)  # Dataset-native phase timeouts.
            write_json(output / 'harbor-process.json', process)
            rows = tb_results(tasks, output / 'harbor')
            write_json(output / 'results.json', rows)
            if process['returncode']:
                raise RuntimeError('Harbor process failed; see harbor.log')
    except (Exception, KeyboardInterrupt) as exc:
        fatal = redact(f'{type(exc).__name__}: {exc}')
        if benchmark == 'tbv2' and (output / 'harbor').exists() and 'tasks' in locals():
            try:
                rows = tb_results(tasks, output / 'harbor')
                write_json(output / 'results.json', rows)
            except (ValueError, KeyError, TypeError):
                pass  # Preserve the original error if a trial artifact is incomplete.
        if (output / 'results.json').exists():
            saved = {r['task_id']: r for r in read_json(output / 'results.json')}
            rows = [saved[r['task_id']] if saved.get(r['task_id'], {}).get('status', 'pending') != 'pending'
                    else r for r in rows]
        for result in rows:
            if result['status'] == 'pending':
                result.update(status='environment_error', detail=fatal)
        write_json(output / 'error.json', {'error': fatal})
    finally:
        oom = docker_oom_events(output, started_at) if rows else None
        for result in rows:
            enrich(result, task_directory(output, benchmark, result['task_id']), output, oom)
        scrub_tree(output)
        summary = summarize(output, rows, time.monotonic() - start, expected, fatal)
    return output, summary
