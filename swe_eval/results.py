import csv
import json
from datetime import datetime
from pathlib import Path
import re
from .config import digest
from .io import read_json, redact, write_json
from .metrics import METRIC_DEFAULTS, INFRA_TIMEOUT_STAGES, archive_index

INFRA = {'api_error', 'environment_error', 'pending'}


def row(task_id, status='pending', detail=''):
    return dict(task_id=task_id, status=status, resolved=None, elapsed_seconds=0,
                input_tokens=None, output_tokens=None, detail=detail,
                **{k: list(v) if isinstance(v, list) else v for k, v in METRIC_DEFAULTS.items()})


FIELDS = list(row('').keys())


def classify(error):
    text = str(error).lower()
    if any(s in text for s in ('verifier', 'rewardfile', 'buildimage', 'docker', 'permission denied')):
        return 'environment_error'
    if any(s in text for s in ('contextwindow', 'contextlength', 'outputlength', 'limitsexceeded',
                               'agenttimeout', 'max_turn', 'max turn', 'max episode')):
        return 'budget_exhausted'
    if any(s in text for s in ('api', 'authentication', 'ratelimit', 'rate limit', 'connection',
                               'modelnotfound', 'unsupportedparams', 'badrequest', 'notfounderror')):
        return 'api_error'
    return 'environment_error'  # Unknown exceptions cannot establish a completed score.


def mini_usage(trajectory):
    usage = [m.get('extra', {}).get('response', {}).get('usage') for m in trajectory.get('messages', [])]
    return {name: sum(u[key] for u in usage if u and isinstance(u.get(key), int))
            if any(u and isinstance(u.get(key), int) for u in usage) else None
            for name, key in [('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens')]}


def swe_result(task_id, agent_dir, scoring_dir, process, oracle=False):
    result = row(task_id)
    result['elapsed_seconds'] = process['elapsed_seconds']
    if process['timed_out']:
        result.update(status='environment_error' if oracle else 'budget_exhausted', detail='Agent wall clock timeout')
        return result
    if process['returncode']:
        result.update(status=classify((Path(agent_dir) / 'agent.log').read_text()), detail='Agent process failed; see agent.log')
        return result
    if not oracle:
        trajectories = list(Path(agent_dir).rglob('*.traj.json'))
        if len(trajectories) != 1:
            result.update(status='environment_error', detail='Missing or duplicate trajectory')
            return result
        trajectory = read_json(trajectories[0])
        result.update(mini_usage(trajectory))
        exit_status = trajectory.get('info', {}).get('exit_status', '')
        if exit_status != 'Submitted':
            result.update(status=classify(exit_status), detail=f'Agent exit: {exit_status}')
            if result['status'] == 'budget_exhausted':
                aggregates = [read_json(p) for p in Path(scoring_dir).glob('*.json')]
                if any(isinstance(a, dict) and task_id in a.get('empty_patch_ids', []) for a in aggregates):
                    result.update(resolved=False, detail=f'Agent exit: {exit_status}; official harness: empty submission')
            return result
    reports = []
    for path in Path(scoring_dir).rglob('report.json'):
        content = read_json(path)
        if task_id in content:
            reports.append(content[task_id])
    if len(reports) != 1 or not isinstance(reports[0].get('resolved'), bool):
        result.update(status='environment_error', detail='Missing or duplicate official harness report')
        if not oracle and not reports:
            # Official harness skips per-instance reports for an empty prediction.
            # The aggregate remains authoritative for this model failure.
            aggregates = [read_json(p) for p in Path(scoring_dir).glob('*.json')]
            if any(isinstance(a, dict) and task_id in a.get('empty_patch_ids', []) for a in aggregates):
                result.update(status='task_failed', resolved=False, detail='Official harness: empty patch')
            else:
                for log in Path(scoring_dir).rglob('run_instance.log'):
                    if log.parent.name == task_id and re.search(
                        r'^.* - INFO - >>>>> Patch Apply Failed:', log.read_text(), re.MULTILINE
                    ):
                        result.update(status='task_failed', resolved=False, detail='Official harness: patch apply failed')
                        break
    else:
        result.update(status='success' if reports[0]['resolved'] else 'task_failed', resolved=reports[0]['resolved'])
    return result


def tb_results(tasks, root):
    results = {task['id']: row(task['id'], 'environment_error', 'Missing Harbor trial result') for task in tasks}
    seen = set()
    for path in Path(root).rglob('result.json'):
        content = read_json(path)
        task_id = content.get('task_name')
        if task_id is None:  # Job aggregate, not a trial.
            continue
        if task_id not in results:
            raise ValueError(f'Unexpected Harbor task: {task_id}')
        if task_id in seen:
            results[task_id] = row(task_id, 'environment_error', 'Duplicate Harbor trial')
            continue
        seen.add(task_id)
        result = row(task_id)
        start, end = content.get('started_at'), content.get('finished_at')
        if start and end:
            result['elapsed_seconds'] = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        agent = content.get('agent_result') or {}
        result.update(input_tokens=agent.get('n_input_tokens'), output_tokens=agent.get('n_output_tokens'))
        error = content.get('exception_info')
        rewards = (content.get('verifier_result') or {}).get('rewards')
        metadata = agent.get('metadata') or {}
        if error:
            detail = error.get('exception_type', '') + ': ' + error.get('exception_message', '')
            result.update(status=classify(detail), detail=detail)
            if result['status'] == 'budget_exhausted' and isinstance(rewards, dict) and isinstance(
                rewards.get('reward'), (int, float)
            ) and end:
                result['resolved'] = rewards['reward'] == 1
        elif not isinstance(rewards, dict) or not isinstance(rewards.get('reward'), (int, float)) or not end:
            result.update(status='environment_error', detail='Missing completed verifier reward')
        else:
            resolved = rewards['reward'] == 1
            result.update(status='success' if resolved else 'task_failed', resolved=resolved)
            limit = content.get('config', {}).get('agent', {}).get('kwargs', {}).get('max_turns')
            if not resolved and limit and metadata.get('n_episodes', 0) >= limit:
                result.update(status='budget_exhausted', detail='Harbor reached configured turn limit')
        results[task_id] = result
    return list(results.values())


def csv_file(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: redact(v) if isinstance(v, str) else
                          json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                          for k, v in r.items()} for r in rows)


def summarize(directory, rows, elapsed, expected_count, fatal=None):
    complete = len(rows) == expected_count and not fatal and all(r['status'] not in INFRA for r in rows)
    success = sum(r['resolved'] is True for r in rows)
    summary = {'schema_version': 2, 'complete': bool(complete), 'scope': 'subset only; not a full benchmark score',
               'success_count': success, 'task_count': expected_count, 'recorded_task_count': len(rows),
               'success_rate': success / expected_count if complete else None,
               'elapsed_seconds': round(elapsed, 3), 'fatal_error': fatal,
               'status_counts': {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})}}
    summary.update(resolved_rate=summary['success_rate'], pass_at_1=summary['success_rate'], attempts_per_task=1,
                   metrics_complete=len(rows) == expected_count and all(r.get('metrics_complete') for r in rows))
    for key in ('input_tokens', 'output_tokens'):
        known = [r[key] for r in rows if r[key] is not None]
        summary[key] = sum(known) if known else None
        summary[key + '_tasks_observed'] = len(known)
    for key in ('agent_rounds', 'tool_rounds', 'tool_call_count'):
        values = [r[key] for r in rows if r.get(key) is not None]
        summary['avg_' + key] = sum(values) / len(values) if values else None
        summary[key + '_tasks_observed'] = len(values)
    for key in ('action_parse_attempts', 'tool_format_error_count', 'unrecoverable_format_error_count',
                'format_auto_correction_count'):
        values = [r[key] for r in rows if r.get(key) is not None]
        summary[key] = sum(values) if values else None
    denominator = summary['action_parse_attempts']
    summary['tool_format_error_rate'] = summary['tool_format_error_count'] / denominator if denominator else None
    for key in ('timeout', 'oom_observed'):
        values = [r[key] for r in rows if r.get(key) is not None]
        summary[key + '_task_count'] = sum(values) if values else None
        summary[key + '_tasks_observed'] = len(values)
        summary[key + '_rate'] = sum(values) / expected_count if len(values) == expected_count else None
    observed_timeout_rows = [r for r in rows if r.get('timeout') is not None]
    for name, stages in [('agent_timeout', {'agent'}), ('tool_timeout', {'tool'}),
                         ('infrastructure_timeout', INFRA_TIMEOUT_STAGES)]:
        count = sum(bool(set(r.get('timeout_stages', [])) & stages) for r in observed_timeout_rows)
        summary[name + '_task_count'] = count if observed_timeout_rows else None
        summary[name + '_rate'] = count / expected_count if len(observed_timeout_rows) == expected_count else None
    summary['oom_observation_scope'] = 'Local Docker state/events and explicit error signals; hosted server internal OOM may be unobservable'
    write_json(Path(directory) / 'results.json', rows)
    write_json(Path(directory) / 'summary.json', summary)
    archive_index(Path(directory), rows)
    csv_file(Path(directory) / 'results.csv', rows, FIELDS)
    csv_file(Path(directory) / 'summary.csv', [{k: v for k, v in summary.items() if not isinstance(v, dict)}],
             [k for k, v in summary.items() if not isinstance(v, dict)])
    return summary


def compare(left, right, output):
    left, right, output = Path(left), Path(right), Path(output)
    a, b = read_json(left / 'manifest.json'), read_json(right / 'manifest.json')
    for manifest in (a, b):
        if digest(manifest['tasks']) != manifest['tasks_sha256']:
            raise ValueError('Frozen task manifest content hash mismatch')
    if a['benchmark'] != b['benchmark'] or a['tasks_sha256'] != b['tasks_sha256']:
        raise ValueError('Comparison requires the same benchmark and identical frozen tasks')
    ca, cb = read_json(left / 'comparison.json'), read_json(right / 'comparison.json')
    if ca != cb:
        differences = [k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)]
        raise ValueError('Incompatible evaluation settings: ' + ', '.join(sorted(differences)))
    sa, sb = read_json(left / 'summary.json'), read_json(right / 'summary.json')
    ra, rb = read_json(left / 'results.json'), read_json(right / 'results.json')
    ids = [t['id'] for t in a['tasks']]
    for records in (ra, rb):
        if len(records) != len(ids) or {r['task_id'] for r in records} != set(ids):
            raise ValueError('Results do not cover the frozen task list exactly once')
    by_a, by_b = {r['task_id']: r for r in ra}, {r['task_id']: r for r in rb}
    changes = []
    for task_id in ids:
        x, y = by_a[task_id], by_b[task_id]
        changes.append({'task_id': task_id, 'baseline_status': x['status'], 'checkpoint_status': y['status'],
                        'baseline_resolved': x['resolved'], 'checkpoint_resolved': y['resolved'],
                        'resolved_delta': int(y['resolved']) - int(x['resolved'])
                        if x['resolved'] is not None and y['resolved'] is not None else None})
        for key in ('tool_rounds', 'tool_call_count', 'tool_format_error_rate', 'output_tokens'):
            changes[-1][key + '_delta'] = y[key] - x[key] if x.get(key) is not None and y.get(key) is not None else None
    complete = sa['complete'] and sb['complete'] and all(r['status'] not in INFRA for r in ra + rb)
    summary = {'complete': complete, 'baseline': str(left), 'checkpoint': str(right),
               'task_count': len(ids), 'success_count_delta': sb['success_count'] - sa['success_count'],
               'success_rate_delta': sb['success_rate'] - sa['success_rate'] if complete else None,
               'elapsed_seconds_delta': sb['elapsed_seconds'] - sa['elapsed_seconds'],
               'scope': 'paired subset regression; not a full benchmark score'}
    for key in ('input_tokens', 'output_tokens'):
        summary[key + '_delta'] = sb[key] - sa[key] if all(
            r[key] is not None for r in ra + rb) else None
    for key in ('resolved_rate', 'pass_at_1', 'avg_tool_rounds', 'avg_tool_call_count',
                'tool_format_error_rate', 'timeout_rate', 'infrastructure_timeout_rate', 'oom_observed_rate'):
        summary[key + '_delta'] = sb[key] - sa[key] if sa.get(key) is not None and sb.get(key) is not None else None
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'comparison.json', summary)
    write_json(output / 'changes.json', changes)
    csv_file(output / 'changes.csv', changes, list(changes[0]))
    csv_file(output / 'comparison.csv', [summary], list(summary))
    return summary
