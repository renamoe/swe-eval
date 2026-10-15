"""Multi-run report: compatibility gate, condition differences and a per-task matrix.

Writes report.json, report.csv and a single offline report.html (inline CSS, no scripts).
Only task IDs, statuses and aggregates are exported; manifests (gold patches) are not.
"""
import html
from pathlib import Path
import time

from .config import digest
from .io import read_json, redact, write_json
from .results import csv_file

SCHEMA_VERSION = 1
# Runs must agree on these; every other recorded condition is listed when it differs.
GATE = ('benchmark', 'tasks_sha256', 'swe_scoring', 'agent_sources_sha256')
GATED_KEYS = {'benchmark', 'tasks_sha256', 'agent_sources_sha256', 'settings.suite.swe_scoring'}


def required(path, name):
    if not (path / name).is_file():
        raise ValueError(f'{path} is not a recorded run (missing {name})')
    return read_json(path / name)


def optional(path):
    return read_json(path) if path.is_file() else {}


def load_run(path):
    path = Path(path).resolve()
    manifest = required(path, 'manifest.json')
    if digest(manifest['tasks']) != manifest['tasks_sha256']:
        raise ValueError(f'{path}: frozen task manifest content hash mismatch')
    results, summary, comparison = (required(path, n) for n in ('results.json', 'summary.json', 'comparison.json'))
    ids = [t['id'] for t in manifest['tasks']]
    if len(results) != len(ids) or {r['task_id'] for r in results} != set(ids):
        raise ValueError(f'{path}: results do not cover the frozen task list exactly once')
    # Runs before schema 3 have no run.json and a flat settings table; missing fields stay empty.
    info, config = optional(path / 'run.json'), optional(path / 'config.json')
    benchmark = manifest['benchmark']
    suite = ((comparison.get('settings') or {}).get('suite')) or {}
    return {'path': path, 'run_id': path.name,
            'model_config': info.get('model_config') or config.get('model_config') or path.name,
            'checkpoint': (config.get('model') or {}).get('checkpoint') or None,
            'regrade': info.get('regrade'), 'benchmark': benchmark, 'manifest': manifest.get('name'), 'ids': ids,
            'gate': {'benchmark': benchmark, 'tasks_sha256': manifest['tasks_sha256'],
                     # Terminal-Bench is always judged by its verifier.
                     'swe_scoring': suite.get('swe_scoring', 'submitted-only') if benchmark == 'swebench' else None,
                     'agent_sources_sha256': comparison.get('agent_sources_sha256')},
            'comparison': comparison, 'summary': summary, 'results': {r['task_id']: r for r in results}}


def find_run(runs_dir, model, suite, benchmark):
    """Latest complete run of a model for a suite and benchmark; else the latest recorded one."""
    candidates = []
    for info_path in Path(runs_dir).glob('*/run.json'):
        info = read_json(info_path)
        if (info.get('model_config'), info.get('suite_config'), info.get('benchmark')) == (model, suite, benchmark) \
                and not info.get('oracle') and not info.get('regrade'):
            candidates.append((info.get('started_at_unix') or 0, info_path.parent))
    recorded = [p for _, p in sorted(candidates, reverse=True)
                if (p / 'summary.json').is_file() and (p / 'manifest.json').is_file()]
    if not recorded:
        raise ValueError(f'No {benchmark} run of model {model} for suite {suite} under {runs_dir}')
    return next((p for p in recorded if read_json(p / 'summary.json').get('complete')), recorded[0])


def collect(refs, suite=None, benchmarks=(), runs_dir=Path('runs')):
    """Group runs by benchmark in argument order; the first run of each group is the baseline."""
    groups = {}
    for ref in refs:
        if Path(ref).is_dir():
            run = load_run(ref)
            if not benchmarks or run['benchmark'] in benchmarks:
                groups.setdefault(run['benchmark'], []).append(run)
            continue
        if suite is None:
            raise ValueError(f'{ref} is not a run directory; model names need --suite')
        for benchmark in benchmarks:
            groups.setdefault(benchmark, []).append(load_run(find_run(runs_dir, ref, suite, benchmark)))
    return groups


def flatten(value, prefix=''):
    if isinstance(value, dict) and value:
        flat = {}
        for key, item in value.items():
            flat.update(flatten(item, f'{prefix}.{key}' if prefix else str(key)))
        return flat
    return {prefix: value}


def short(value):
    text = str(value)
    return text[:12] if len(text) == 64 else text  # Abbreviate SHA-256 digests in messages.


def check_gate(runs):
    base = runs[0]
    for run in runs[1:]:
        bad = [f'{k}: {short(base["gate"][k])} vs {short(run["gate"][k])}'
               for k in GATE if run['gate'][k] != base['gate'][k]]
        if bad:
            raise ValueError(f'{run["run_id"]} cannot be compared with {base["run_id"]} ({"; ".join(bad)})')


def differences(runs):
    """Recorded conditions that differ between runs; they do not block the report."""
    flat = [flatten(r['comparison']) for r in runs]
    keys = sorted(set().union(*flat) - GATED_KEYS)
    return [{'key': k, 'values': [f.get(k) for f in flat]} for k in keys
            if any(f.get(k) != flat[0].get(k) for f in flat[1:])]


def labels(runs):
    names = [r['model_config'] for r in runs]
    return [r['run_id'] if names.count(n) > 1 else n for r, n in zip(runs, names)]


def change(base, other):
    if base is None or other is None:
        return None
    return 'gained' if other and not base else 'regressed' if base and not other else 'same'


def section(runs):
    check_gate(runs)
    names = labels(runs)
    ids = runs[0]['ids']
    base = runs[0]
    models = []
    for name, run in zip(names, runs):
        summary, rows = run['summary'], list(run['results'].values())
        scored = [r for r in rows if r['resolved'] is not None]
        paired = [i for i in ids if base['results'][i]['resolved'] is not None
                  and run['results'][i]['resolved'] is not None]
        changes = [change(base['results'][i]['resolved'], run['results'][i]['resolved']) for i in paired]
        complete = bool(summary.get('complete'))
        both = complete and bool(base['summary'].get('complete'))
        models.append({
            'label': name, 'run_id': run['run_id'], 'model_config': run['model_config'],
            'checkpoint': run['checkpoint'], 'regrade': run['regrade'], 'complete': complete,
            'stopped': summary.get('stopped'), 'task_count': len(ids), 'scored_count': len(scored),
            'passed_count': sum(r['resolved'] is True for r in rows),
            'success_rate': summary.get('success_rate') if complete else None,
            'input_tokens': summary.get('input_tokens'), 'output_tokens': summary.get('output_tokens'),
            'avg_agent_rounds': summary.get('avg_agent_rounds'), 'elapsed_seconds': summary.get('elapsed_seconds'),
            'termination_counts': summary.get('termination_counts') or {},
            'status_counts': summary.get('status_counts') or {},
            'paired_count': len(paired), 'gained': changes.count('gained'), 'regressed': changes.count('regressed'),
            # A rate difference is only meaningful when both runs scored every task.
            'success_rate_delta': round(summary['success_rate'] - base['summary']['success_rate'], 6)
            if both else None})
    matrix = []
    for task_id in ids:
        cells = []
        for run in runs:
            result = run['results'][task_id]
            cells.append({'status': result['status'], 'resolved': result['resolved'],
                          'scored_from': result.get('scored_from'), 'termination': result.get('termination'),
                          'change': change(base['results'][task_id]['resolved'], result['resolved'])
                          if run is not base else None})
        matrix.append({'task_id': task_id, 'runs': cells})
    return {'benchmark': base['benchmark'], 'manifest': base['manifest'], 'baseline': names[0],
            'gate': base['gate'], 'differences': differences(runs), 'models': models, 'matrix': matrix}


def csv_rows(report):
    rows = []
    for part in report['benchmarks']:
        for task in part['matrix']:
            for model, cell in zip(part['models'], task['runs']):
                rows.append({'benchmark': part['benchmark'], 'task_id': task['task_id'], 'model': model['label'],
                             'run_id': model['run_id'], **cell})
    return rows


def build(refs, suite=None, benchmarks=('swebench', 'tbv2'), runs_dir=Path('runs'), single=False):
    groups = collect(refs, suite, benchmarks, runs_dir)
    if not groups:
        raise ValueError('No runs to report')
    if single and len(groups) > 1:
        raise ValueError('Comparison requires runs of the same benchmark')
    sections = [section(runs) for _, runs in sorted(groups.items())]
    return {'schema_version': SCHEMA_VERSION, 'suite': suite,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'scope': 'paired subset regression; not a full benchmark score', 'benchmarks': sections}


def report(refs, output, suite=None, benchmarks=('swebench', 'tbv2'), runs_dir=Path('runs'), single=False):
    result = build(refs, suite, benchmarks, runs_dir, single)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'report.json', result)
    rows = csv_rows(result)
    csv_file(output / 'report.csv', rows, ['benchmark', 'task_id', 'model', 'run_id', 'status', 'resolved',
                                           'scored_from', 'termination', 'change'])
    (output / 'report.html').write_text(redact(render(result)))
    return result


CSS = '''body{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1b1f24;background:#fff}
h1{font-size:20px}h2{font-size:17px;margin-top:32px}table{border-collapse:collapse;margin:8px 0 16px}
th,td{border:1px solid #d0d7de;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f6f8fa}
td.num{text-align:right;font-variant-numeric:tabular-nums}.pass{background:#dafbe1}.fail{background:#ffebe9}
.none{background:#f6f8fa;color:#57606a}.gained{outline:2px solid #1a7f37}.regressed{outline:2px solid #cf222e}
.warn{border-left:4px solid #bf8700;background:#fff8c5;padding:8px 12px}.note{color:#57606a}
code{font-family:ui-monospace,monospace;font-size:12px}'''


def cell(value):
    if value is None:
        return '<td class="none">–</td>'
    if isinstance(value, float):
        return f'<td class="num">{value:.4g}</td>'
    if isinstance(value, int) and not isinstance(value, bool):
        return f'<td class="num">{value}</td>'
    return f'<td>{html.escape(str(value))}</td>'


def table(header, rows):
    head = ''.join(f'<th>{html.escape(h)}</th>' for h in header)
    return f'<table><tr>{head}</tr>' + ''.join(f'<tr>{"".join(r)}</tr>' for r in rows) + '</table>'


def rate(value):
    return None if value is None else f'{value:.1%}'


def counts(values):
    return ', '.join(f'{k} {v}' for k, v in values.items()) or None


def render(report):
    escape = html.escape
    parts = [f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>swe-eval report</title>'
             f'<style>{CSS}</style><h1>swe-eval report{" · " + escape(report["suite"]) if report["suite"] else ""}</h1>'
             f'<p class="note">{escape(report["generated_at"])} · {escape(report["scope"])}</p>']
    for part in report['benchmarks']:
        parts.append(f'<h2>{escape(part["benchmark"])} · {escape(str(part["manifest"]))}'
                     f' · baseline {escape(part["baseline"])}</h2>')
        if part['differences']:
            rows = [[f'<td><code>{escape(d["key"])}</code></td>', *(cell(v) for v in d['values'])]
                    for d in part['differences']]
            parts.append('<div class="warn">评测条件存在差异，不是受控对比：'
                         + table(['条件', *(m['label'] for m in part['models'])], rows) + '</div>')
        rows = []
        for m in part['models']:
            rows.append([cell(m['label']), cell('完整' if m['complete'] else '不完整'),
                         cell(f'{m["passed_count"]}/{m["scored_count"]}/{m["task_count"]}'),
                         cell(rate(m['success_rate'])), cell(rate(m['success_rate_delta'])),
                         cell(m['gained'] if m is not part['models'][0] else None),
                         cell(m['regressed'] if m is not part['models'][0] else None),
                         cell(m['paired_count']), cell(m['input_tokens']), cell(m['output_tokens']),
                         cell(m['avg_agent_rounds']), cell(m['elapsed_seconds']),
                         cell(counts(m['termination_counts'])), cell(counts(m['status_counts']))])
        parts.append(table(['模型', '状态', '通过/已评分/计划', '成功率', '差值', '新增通过', '回退', '配对题数',
                            '输入 token', '输出 token', '平均轮数', '耗时 s', '终止原因', '状态'], rows))
        if any(not m['complete'] for m in part['models']):
            parts.append('<p class="note">不完整 run 只比较双方都已评分的题，不给成功率差值。</p>')
        rows = []
        for task in part['matrix']:
            line = [f'<td><code>{escape(task["task_id"])}</code></td>']
            for c in task['runs']:
                kind = 'none' if c['resolved'] is None else 'pass' if c['resolved'] else 'fail'
                mark = {'gained': ' ↑', 'regressed': ' ↓'}.get(c['change'], '')
                label = ('通过' if c['resolved'] else '未通过') if c['resolved'] is not None else c['status']
                if c['scored_from']:
                    label += f' ({c["scored_from"]})'
                title = escape(f'{c["status"]}; termination={c["termination"]}')
                line.append(f'<td class="{kind} {c["change"] or ""}" title="{title}">{escape(label + mark)}</td>')
            rows.append(line)
        parts.append(table(['任务', *(m['label'] for m in part['models'])], rows))
    return '\n'.join(parts) + '\n</html>\n'
