"""Metrics derived from observed native agent events and official trial artifacts."""
import json
from pathlib import Path
import re
import subprocess
import time

from .io import read_json, write_json

METRIC_DEFAULTS = {
    'agent_rounds': None, 'llm_call_count': None, 'tool_rounds': None, 'tool_call_count': None,
    'action_parse_attempts': None, 'tool_format_error_count': None, 'tool_format_error_rate': None,
    'unrecoverable_format_error_count': None, 'format_auto_correction_count': None,
    'timeout': None, 'timeout_stages': [], 'oom_observed': None,
    'termination': None, 'reasoning_tokens': None, 'max_prompt_tokens': None, 'max_completion_tokens': None,
    'context_peak': None, 'length_finish_count': None, 'llm_seconds': None, 'requests_path': None,
    'token_usage_complete': None, 'metrics_complete': False, 'trace_complete': False,
    'trajectory_paths': [], 'events_paths': [], 'scoring_paths': [], 'metrics_notes': [],
}
TIMEOUT = re.compile(r'timeout|timed out|time limit', re.I)
OOM = re.compile(r'\boomkilled\b|\bout[ -]of[ -]memory\b|\boom(?:error|kill)?\b', re.I)
INFRA_TIMEOUT_STAGES = {'api', 'agent_setup', 'environment_setup', 'verifier', 'unknown'}
TERMINATIONS = ('submitted', 'turn_limit', 'timeout', 'context_limit', 'output_truncated',
                'api_error', 'env_error', 'stopped')
# Agent-side causes, checked in order against the result detail (native exception names and our labels).
CAUSES = [('timeout', ('agent wall clock timeout', 'agenttimeout')),
          ('context_limit', ('contextwindow', 'contextlength')),
          ('output_truncated', ('outputlength',)),
          ('turn_limit', ('limitsexceeded', 'turn limit', 'max_turn', 'max turn', 'max episode'))]


def termination(result, observed):
    """How the agent run ended; separate from the score and from the INFRA status."""
    if result['status'] == 'stopped':
        return 'stopped'
    detail = result.get('detail', '').lower()
    for name, needles in CAUSES:
        if any(n in detail for n in needles):
            return name
    if result['status'] == 'api_error':
        return 'api_error'
    if result['status'] in ('environment_error', 'pending'):
        return 'env_error'
    # Gold/oracle runs have no agent telemetry and no agent termination.
    return 'submitted' if observed and result['status'] in ('success', 'task_failed') else None


def request_records(events):
    """One record per model request: usage, finish reason and latency."""
    starts = {e['response_index']: e['timestamp'] for e in events if e['event'] == 'llm_start'}
    raw = {e['response_index']: e for e in events if e['event'] == 'llm_raw'}
    records = []
    for event in events:
        if event['event'] not in ('llm_response', 'llm_error'):
            continue
        index = event['response_index']
        source = raw.get(index)
        if source is None and event['event'] == 'llm_response':
            response = event.get('response') or {}
            native = (response.get('extra') or {}).get('response') or {}  # mini-SWE-agent keeps the raw body.
            source = {'finish_reason': ((native.get('choices') or [{}])[0] or {}).get('finish_reason'),
                      'usage': native.get('usage') or response.get('usage')}
        usage = (source or {}).get('usage') or {}
        details = usage.get('completion_tokens_details') or {}
        start = starts.get(index)
        records.append({
            'index': index, 'ok': event['event'] == 'llm_response',
            'finish_reason': (source or {}).get('finish_reason'),
            'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
            'reasoning_tokens': details.get('reasoning_tokens'),
            'seconds': round(event['timestamp'] - start, 3) if start is not None else None,
            'error': event.get('exception_type') if event['event'] == 'llm_error' else None})
    return records


def known(values):
    return [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]


def request_metrics(result, records, budget):
    prompts = known(r['prompt_tokens'] for r in records)
    completions = known(r['completion_tokens'] for r in records)
    reasoning = known(r['reasoning_tokens'] for r in records)
    seconds = known(r['seconds'] for r in records)
    result.update(max_prompt_tokens=max(prompts) if prompts else None,
                  max_completion_tokens=max(completions) if completions else None,
                  reasoning_tokens=sum(reasoning) if reasoning else None,
                  length_finish_count=sum(r['finish_reason'] == 'length' for r in records) if records else None,
                  llm_seconds=round(sum(seconds), 3) if seconds else None)
    # Share of the input budget (context_length - max_tokens) used by the largest prompt.
    if prompts and budget:
        result['context_peak'] = round(max(prompts) / budget, 4)


def docker_oom_events(output, started_at):
    """Read bounded Docker OOM events, including destroyed verifier containers."""
    prefixes = []
    for path in (output / 'harbor').rglob('result.json'):
        result = read_json(path)
        if result.get('trial_name'):
            prefixes.append((result['task_name'], result['trial_name'].lower()))
    for path in output.rglob('events.jsonl'):
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event['event'] in ('container_state', 'container_start', 'container_project'):
                task_id = event.get('task_id') or (path.parent.parent.name.rsplit('__', 1)[0]
                                                   if path.parent.name == 'agent' else path.parent.name)
                prefix = event.get('container_id') or event.get('project')
                if prefix:
                    prefixes.append((task_id, prefix.lower()))
    observed = {'available': False, 'task_ids': [], 'scope': 'local Docker OOM events only; hosted API server OOM may be unobservable'}
    try:
        proc = subprocess.run(['docker', 'events', '--since', str(int(started_at)),
                               '--until', str(int(time.time())), '--filter', 'event=oom',
                               '--format', '{{json .}}'], capture_output=True, text=True, timeout=10)
        if proc.returncode:
            observed['error'] = proc.stderr[-1000:]
        else:
            observed['available'] = True
            ids = set()
            for line in proc.stdout.splitlines():
                event = json.loads(line)
                actor = event.get('Actor', {})
                labels = actor.get('Attributes', {})
                identity = ' '.join([actor.get('ID', ''), labels.get('name', ''),
                                     labels.get('com.docker.compose.project', '')]).lower()
                ids.update(task_id for task_id, prefix in prefixes if prefix in identity)
            observed['task_ids'] = sorted(ids)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        observed['error'] = str(exc)
    write_json(output / 'docker-oom-events.json', observed)
    return observed


def task_directory(output, benchmark, task_id):
    if benchmark == 'swebench':
        return output / 'tasks' / task_id
    matches = [p for p in (output / 'harbor' / 'job').glob(task_id + '__*') if p.is_dir()]
    return matches[0] if len(matches) == 1 else output / 'missing-trial' / task_id


def enrich(result, directory, output, oom_events=None, budget=None):
    defaults = {k: list(v) if isinstance(v, list) else v for k, v in METRIC_DEFAULTS.items()}
    result.update(defaults)
    trace_paths = sorted([*directory.rglob('*.traj.json'), *directory.rglob('trajectory.json'),
                          *directory.rglob('messages.json'), *directory.rglob('terminus_2.pane'),
                          *directory.rglob('*.cast')])
    event_paths = sorted(directory.rglob('events.jsonl'))
    result['trajectory_paths'] = [str(p.relative_to(output)) for p in trace_paths]
    result['events_paths'] = [str(p.relative_to(output)) for p in event_paths]
    result['scoring_paths'] = [str(p.relative_to(output)) for p in sorted(directory.rglob('*'))
                               if p.is_file() and not p.is_symlink() and p.name in {
                                   'report.json', 'run_instance.log', 'test_output.txt', 'test-stdout.txt',
                                   'reward.txt', 'trial.log', 'exception.txt', 'harness.log'}]
    events, partial = [], False
    if len(event_paths) == 1:
        for line in event_paths[0].read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                partial = True
                result['metrics_notes'].append('Incomplete telemetry line; counters are lower bounds')
    elif len(event_paths) > 1:
        result['metrics_notes'].append('Multiple event streams; metrics are ambiguous')
    if events:
        starts = [e for e in events if e['event'] == 'llm_start']
        responses = [e for e in events if e['event'] == 'llm_response']
        tools = [e for e in events if e['event'] == 'tool_start']
        parses = [e for e in events if e['event'] == 'action_parse']
        errors = sum(bool(e.get('raw_error')) for e in parses)
        result.update(agent_rounds=len(responses), llm_call_count=len(starts),
                      tool_rounds=len({e['response_index'] for e in tools}), tool_call_count=len(tools),
                      action_parse_attempts=len(parses), tool_format_error_count=errors,
                      tool_format_error_rate=errors / len(parses) if parses else None,
                      unrecoverable_format_error_count=sum(bool(e.get('error')) for e in parses),
                      format_auto_correction_count=sum(bool(e.get('raw_error')) and not e.get('error') for e in parses))
        usages = []
        for event in responses:
            response = event['response']
            usage = response.get('usage') or (response.get('extra') or {}).get('response', {}).get('usage')
            usages.append(usage or {})
        for field, key in [('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens')]:
            values = [u[key] for u in usages if isinstance(u.get(key), int)]
            if values:
                # Native totals may include additional calls from error recovery.
                result[field] = max(result.get(field) or 0, sum(values))
        result['token_usage_complete'] = bool(responses) and len(responses) == len(starts) and all(
            isinstance(u.get('prompt_tokens'), int) and isinstance(u.get('completion_tokens'), int) for u in usages)
        records = request_records(events)
        path = event_paths[0].parent / 'requests.json'
        write_json(path, records)
        result['requests_path'] = str(path.relative_to(output))
        request_metrics(result, records, budget)
        result['trace_complete'] = not partial and any(e['event'] == 'agent_end' for e in events)
        result['metrics_complete'] = result['trace_complete'] and result['token_usage_complete']
        if not result['trace_complete']:
            result['metrics_notes'].append('Agent interrupted; all persisted events retained, in-flight response may be unavailable')
    else:
        result['metrics_notes'].append('No agent telemetry; counters unavailable, not assumed zero')

    stages = set()
    for event in events:
        if event['event'] not in ('llm_error', 'tool_error', 'agent_error'):
            continue
        detail = event.get('exception_type', '') + ' ' + event.get('detail', '')
        if TIMEOUT.search(detail):
            stages.add({'llm_error': 'api', 'tool_error': 'tool', 'agent_error': 'agent'}[event['event']])
    detail = result.get('detail', '')
    if TIMEOUT.search(detail):
        lower = detail.lower()
        if 'agenttimeout' in lower or 'agent wall clock' in lower:
            stages.add('agent')
        elif 'setup' in lower:
            stages.add('agent_setup')
        elif 'verifier' in lower or 'harness' in lower:
            stages.add('verifier')
        elif result['status'] == 'api_error':
            stages.add('api')
        else:
            stages.add('unknown')
    score = directory / 'score-process.json'
    if score.exists() and read_json(score)['timed_out']:
        stages.add('verifier')
    # Native Harbor trial timestamps identify setup failures separately from the agent budget.
    trial = directory / 'result.json'
    if trial.exists():
        data = read_json(trial)
        exception = data.get('exception_info') or {}
        message = exception.get('exception_type', '') + ' ' + exception.get('exception_message', '')
        if TIMEOUT.search(message) and not data.get('agent_execution'):
            stages.discard('unknown')
            stages.add('agent_setup' if data.get('agent_setup') else 'environment_setup')
    if events or stages:
        result.update(timeout=bool(stages), timeout_stages=sorted(stages))
    explicit_errors = [detail] + [e.get('exception_type', '') + ' ' + e.get('detail', '')
                                  for e in events if e['event'].endswith('_error')]
    probes = [e for e in events if e['event'] == 'container_state']
    oom = any(e.get('oom_killed') is True for e in probes) or any(OOM.search(s) for s in explicit_errors)
    if oom_events and result['task_id'] in oom_events.get('task_ids', []):
        oom = True
    if oom or probes or (oom_events and oom_events['available']):
        result['oom_observed'] = bool(oom)
    if oom:
        result.update(status='environment_error', detail=detail + '; observed OOM', resolved=None)
    result['termination'] = termination(result, bool(events))
    result['metrics_complete'] = bool(result['metrics_complete'] and result['timeout'] is not None
                                      and result['oom_observed'] is not None)
    return result


def archive_index(output, rows):
    failures = []
    for result in rows:
        if result['resolved'] is True and result['status'] == 'success':
            continue
        failures.append({k: result[k] for k in ('task_id', 'status', 'detail', 'trace_complete',
                                               'trajectory_paths', 'events_paths', 'metrics_notes')})
        failures[-1]['scoring_paths'] = result['scoring_paths']
    write_json(output / 'failures.json', failures)
