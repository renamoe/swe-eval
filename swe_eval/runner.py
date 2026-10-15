import hashlib
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

from .config import ROOT, base_env, digest
from .doctor import doctor
from .io import execute, read_json, redact, scrub_tree, write_json
from .results import candidate_result, row, summarize, swe_result, tb_results
from .metrics import docker_oom_events, enrich, task_directory


def environment_python(benchmark):
    project = ROOT / 'environments' / benchmark
    if not (project / 'uv.lock').exists() or not (project / '.venv' / 'bin' / 'python').exists():
        raise RuntimeError(f'{benchmark} environment is not locked/installed; run bash scripts/setup.sh lock, then bash scripts/setup.sh sync')
    return str(project / '.venv' / 'bin' / 'python')


# Manifest names supported until suite-defined sampling lands (M3).
MANIFEST_SIZES = {'smoke': 1, 'small': 10}


RUN_LABEL = 'swe_eval.run'
AGENT_GRACE = 60     # Lets the agent worker save trajectory and candidate.patch.
HARBOR_GRACE = 120   # Lets Harbor stop its compose environments after cancellation.


def docker_lines(*args):
    try:
        process = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return process.stdout.split('\n') if process.returncode == 0 else None


def sanitize_project(name):
    # Same rule as Harbor's _sanitize_docker_compose_project_name.
    name = name.lower()
    return re.sub(r'[^a-z0-9_-]', '-', name if re.match(r'^[a-z0-9]', name) else '0' + name)


def cleanup_containers(output, run_id):
    """Remove containers and compose networks left by this run only."""
    containers, networks = set(), set()
    found = docker_lines('ps', '-aq', '--filter', f'label={RUN_LABEL}={run_id}')
    containers.update(found or [])
    # SWE harness names its containers sweb.eval.<instance>.<harness run_id>.
    for line in docker_lines('ps', '-a', '--filter', f'name=.{run_id}', '--format', '{{.ID}} {{.Names}}') or []:
        identifier, _, name = line.partition(' ')
        if name.startswith('sweb.eval.') and name.endswith('.' + run_id):
            containers.add(identifier)
    # Harbor compose projects start with the sanitized trial directory name.
    trials = [sanitize_project(p.name) for p in (output / 'harbor' / 'job').glob('*__*') if p.is_dir()]
    if trials:
        for kind, target in (('ps', containers), ('network', networks)):
            command = ['ps', '-a'] if kind == 'ps' else ['network', 'ls']
            for line in docker_lines(*command, '--format', '{{.ID}} {{.Label "com.docker.compose.project"}}') or []:
                identifier, _, project = line.partition(' ')
                if project and any(project.startswith(t) for t in trials):
                    target.add(identifier)
    containers.discard('')
    removed = {'containers': sorted(containers), 'networks': sorted(networks)}
    if containers:
        docker_lines('rm', '-f', *sorted(containers))
    if networks:
        docker_lines('network', 'rm', *networks)
    left = docker_lines('ps', '-aq', '--filter', f'label={RUN_LABEL}={run_id}')
    removed['remaining_labeled'] = None if left is None else [x for x in left if x]
    write_json(output / 'cleanup.json', removed)
    return removed


class TrialWatcher(threading.Thread):
    """Print one line per finished Harbor trial while the single job runs."""

    def __init__(self, root, total):
        super().__init__(daemon=True)
        self.root, self.total, self.seen, self.done = root, total, set(), threading.Event()

    def run(self):
        while not self.done.wait(5):
            self.scan()

    def scan(self):
        for path in sorted(self.root.glob('job/*/result.json')):
            if path in self.seen:
                continue
            try:
                content = read_json(path)
            except (OSError, ValueError):
                continue
            if not content.get('finished_at'):
                continue
            self.seen.add(path)
            result = tb_results([{'id': content.get('task_name')}], path.parent)[0]
            progress(len(self.seen), self.total, result)


def progress(index, total, result):
    print(f"[{index}/{total}] {result['task_id']}: {result['status']} ({result['elapsed_seconds']:.0f}s)", flush=True)


def worker(benchmark, args, log, cwd=ROOT, timeout=300):
    # Helper workers never call the model; they get no credentials.
    result = execute([environment_python(benchmark), '-m', 'swe_eval.worker', *map(str, args)],
                     cwd, log, env=base_env(), timeout=timeout)
    if result['timed_out'] or result['returncode']:
        raise RuntimeError(f'Worker {args[0]} failed; see {log}')
    return result


def validate_manifest(manifest, benchmark, profile):
    n = MANIFEST_SIZES[profile]
    tasks = manifest['tasks']
    ids = [t['id'] for t in tasks]
    if manifest['benchmark'] != benchmark or manifest['profile'] != profile or len(ids) != n:
        raise ValueError('Manifest benchmark, profile or task count mismatch')
    if ids != sorted(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', x) or x in ('.', '..') for x in ids):
        raise ValueError('Manifest IDs must be unique, sorted and safe path components')
    if manifest['tasks_sha256'] != digest(tasks):
        raise ValueError('Manifest content hash mismatch')
    return manifest


def prepare(benchmark, profile):
    if profile not in MANIFEST_SIZES:
        raise ValueError(f'Unsupported manifest {profile}; expected one of {", ".join(MANIFEST_SIZES)}')
    path = ROOT / 'profiles' / f'{benchmark}-{profile}.json'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use an exclusive marker to prevent competing preparation overwrites.
        marker = path.with_suffix('.preparing')
        try:
            with marker.open('x'):
                temporary = path.with_suffix('.candidate.json')
                try:
                    worker(benchmark, ['prepare', benchmark, profile, temporary],
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
        'kwargs': {'api_base': config.base_url, 'max_turns': config.max_turns, 'parser_name': 'json',
                   'temperature': config.sampling_params['temperature'], 'interleaved_thinking': False,
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


def score_swe(config, task_id, directory, predictions, label):
    """Run the official harness on one prediction; label is the harness run_id (names its containers)."""
    scoring = directory / 'scoring'
    write_json(directory / 'predictions.json', predictions)
    process = execute([environment_python('swebench'), '-m', 'swebench.harness.run_evaluation',
                       '--dataset_name', str(directory / 'dataset.json'), '--split', 'test',
                       '--predictions_path', str(directory / 'predictions.json'),
                       '--instance_ids', task_id, '--max_workers', '1',
                       '--run_id', label, '--timeout', str(config.swe_score_timeout)],
                      scoring, scoring / 'harness.log', base_env(), config.swe_score_timeout)
    write_json(directory / 'score-process.json', process)
    return process


def run_swe(config, tasks, output, oracle, rows):
    """Fill rows in place, one task at a time, so an interrupt keeps finished results."""
    run_id = output.name
    if not oracle:
        worker('swebench', ['swe-config', output / 'config.json', output / 'agent-config.json', run_id],
               output / 'config.log')
    for index, task in enumerate(tasks):
        task_id = task['id']
        print(f'[{index + 1}/{len(tasks)}] {task_id}: running', flush=True)
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
                              ROOT, agent_dir / 'agent.log', config.child_env(), config.swe_agent_timeout,
                              grace=AGENT_GRACE)
        write_json(directory / 'agent-process.json', process)
        before = swe_result(task_id, agent_dir, scoring, process, oracle)
        # Harness verdict needed: formal submission, oracle, or the empty-submission budget case.
        budget_submission = (before['status'] == 'budget_exhausted' and not process['timed_out']
                             and (agent_dir / 'preds.json').exists())
        candidate = agent_dir / 'candidate.patch'
        grade_candidate = (not oracle and config.swe_scoring == 'final-workspace'
                           and before['status'] == 'budget_exhausted'
                           and candidate.exists() and candidate.read_text().strip())
        if grade_candidate:
            score_process = score_swe(config, task_id, directory, {task_id: {
                'instance_id': task_id, 'model_name_or_path': 'candidate',
                'model_patch': candidate.read_text()}}, run_id)
            result = candidate_result(before, task_id, scoring)
        elif not oracle and before['detail'] != 'Missing or duplicate official harness report' and not budget_submission:
            result, score_process = before, None
        else:
            predictions = read_json(agent_dir / 'preds.json')
            # Safe, stable harness label. Actual model/checkpoint is recorded in config.json.
            predictions[task_id]['model_name_or_path'] = 'gold' if oracle else 'candidate'
            score_process = score_swe(config, task_id, directory, predictions, run_id)
            result = swe_result(task_id, agent_dir, scoring, process, oracle)
            if budget_submission and result['resolved'] is None and not score_process['returncode'] \
                    and not score_process['timed_out']:
                result.update(status='environment_error', detail='Missing official harness outcome for exhausted budget')
        if score_process and (score_process['returncode'] or score_process['timed_out']):
            result.update(status='environment_error', resolved=None,
                          detail='Official harness timed out; see harness.log' if score_process['timed_out']
                          else 'Official harness process failed; see harness.log')
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        rows[index] = result
        write_json(output / 'results.json', rows)
        progress(index + 1, len(tasks), result)
    return rows


def run_suite(config, runs_dir=Path('runs'), run_id=None, benchmarks=None, oracle=False):
    """Run each benchmark of the suite in order; an interrupt stops the remaining ones."""
    benchmarks = benchmarks or list(config.benchmarks)
    if run_id and len(benchmarks) > 1:
        raise ValueError('--run-id needs a single benchmark')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    results = []
    for benchmark in benchmarks:
        name = run_id or f'{config.model_config}__{config.suite_config}__{benchmark}__{stamp}'
        output, summary = run(benchmark, name, config, runs_dir, oracle)
        results.append({'benchmark': benchmark, 'output': str(output), **summary})
        if summary.get('stopped'):
            break
    return results


def run(benchmark, run_id, config, runs_dir=Path('runs'), oracle=False):
    if benchmark not in config.benchmarks:
        raise ValueError(f'Suite {config.suite_config} does not include {benchmark}')
    profile = config.manifests[benchmark]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('run-id must be a safe directory name')
    output = (runs_dir / run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    started_at = time.time()
    expected = MANIFEST_SIZES.get(profile, 0)
    rows, fatal, stopped = [], None, False
    write_json(output / 'config.json', config.snapshot())
    write_json(output / 'run.json', {'benchmark': benchmark, 'profile': profile, 'run_id': run_id,
                                   'model_config': config.model_config, 'suite_config': config.suite_config,
                                   'oracle': oracle, 'started_at_unix': time.time()})
    print(f'Artifacts: {output}', flush=True)
    try:
        manifest = prepare(benchmark, profile)
        tasks = manifest['tasks']
        write_json(output / 'manifest.json', manifest)
        rows = [row(t['id']) for t in tasks]
        write_json(output / 'results.json', rows)
        worker(benchmark, ['metadata', benchmark, output / 'dependencies.json'], output / 'metadata.log')
        lock = ROOT / 'environments' / benchmark / 'uv.lock'
        (output / 'environment.uv.lock').write_bytes(lock.read_bytes())
        metadata = read_json(output / 'dependencies.json')
        write_json(output / 'comparison.json', {
            'benchmark': benchmark, 'tasks_sha256': manifest['tasks_sha256'],
            'settings': config.comparison_settings(), 'oracle': oracle, 'concurrency': 1, 'attempts': 1,
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
            run_swe(config, tasks, output, oracle, rows)
        else:
            write_json(output / 'harbor-config.json', tb_config(config, tasks, output, oracle))
            watcher = TrialWatcher(output / 'harbor', len(tasks))
            watcher.start()
            try:
                process = execute([environment_python(benchmark), '-m', 'swe_eval.worker', 'tb-run',
                                   str(output / 'harbor-config.json')], output, output / 'harbor.log',
                                  config.child_env(api=not oracle), timeout=None,  # Dataset-native phase timeouts.
                                  grace=HARBOR_GRACE)
            finally:
                watcher.done.set()
            watcher.scan()
            write_json(output / 'harbor-process.json', process)
            rows = tb_results(tasks, output / 'harbor')
            write_json(output / 'results.json', rows)
            if process['returncode']:
                raise RuntimeError('Harbor process failed; see harbor.log')
    except (Exception, KeyboardInterrupt) as exc:
        stopped = isinstance(exc, KeyboardInterrupt)
        fatal = 'Stopped by user' if stopped else redact(f'{type(exc).__name__}: {exc}')
        if stopped:
            print('Stopping: terminated the current task; cleaning up containers', flush=True)
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
            if result['status'] == 'pending' or stopped and interrupted(result):
                result.update(status='stopped' if stopped else 'environment_error', resolved=None,
                              detail=fatal if result['status'] == 'pending' else f"{fatal}; {result['detail']}")
        write_json(output / 'error.json', {'error': fatal})
    finally:
        # A second Ctrl-C must not abort cleanup and leave containers behind.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            cleanup_containers(output, run_id)
            oom = docker_oom_events(output, started_at) if rows else None
            for result in rows:
                enrich(result, task_directory(output, benchmark, result['task_id']), output, oom)
            scrub_tree(output)
            summary = summarize(output, rows, time.monotonic() - start, expected, fatal)
            summary['stopped'] = stopped
            write_json(output / 'summary.json', summary)
        finally:
            signal.signal(signal.SIGINT, previous)
    return output, summary


def interrupted(result):
    # Harbor trials in flight or not started when the user stopped (SWE in-flight rows are still pending).
    return result['resolved'] is None and ('Missing Harbor trial result' in result['detail']
                                           or 'CancelledError' in result['detail'])
