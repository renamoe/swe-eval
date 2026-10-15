"""Read-only local web view: run list, run and task details, and the section 7 report.

Standard library http.server, one inline page polling JSON every 3 seconds. Files are served only
from registered runs under runs/, as text/plain; gold patches (manifests, task records) never are.
"""
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, urlsplit

from .io import read_json, redact
from .metrics import task_directory
from .report import build, render

SAFE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
# Files that hold gold patches or full task records.
HIDDEN = {'manifest.json', 'task.json', 'dataset.json'}
FILE_LIMIT = 2_000_000
TEXT_LIMIT = 20_000
ROW_FIELDS = ('task_id', 'status', 'resolved', 'scored_from', 'termination', 'elapsed_seconds',
              'input_tokens', 'output_tokens', 'agent_rounds', 'max_prompt_tokens', 'context_peak', 'detail')


def load(path, default=None):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return default


def alive(pid):
    if not pid:
        return False
    try:
        return b'swe_eval' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return False


def state(path, info, summary):
    if summary is not None:
        return 'complete' if summary.get('complete') else 'stopped' if summary.get('stopped') else 'incomplete'
    if alive(info.get('pid')):
        return 'running'
    return 'dead' if info.get('pid') else 'unknown'  # No summary and the process is gone.


def events(path):
    out = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # A line being written right now.
    return out


def usage_of(event):
    response = event.get('response') or {}
    return response.get('usage') or ((response.get('extra') or {}).get('response') or {}).get('usage') or {}


def live(stream):
    """Counters from an event stream in progress; final rows get the full metrics at the end of a run."""
    responses = [e for e in stream if e['event'] == 'llm_response']
    usages = [usage_of(e) for e in responses]
    prompts = [u['prompt_tokens'] for u in usages if isinstance(u.get('prompt_tokens'), int)]
    completions = [u['completion_tokens'] for u in usages if isinstance(u.get('completion_tokens'), int)]
    return {'agent_rounds': len(responses), 'input_tokens': sum(prompts) if prompts else None,
            'output_tokens': sum(completions) if completions else None,
            'max_prompt_tokens': max(prompts) if prompts else None,
            'last_event_at': stream[-1]['timestamp'] if stream else None}


def event_file(directory):
    found = sorted(directory.rglob('events.jsonl')) if directory.is_dir() else []
    return found[0] if len(found) == 1 else None


class Runs:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def path(self, run_id):
        if not SAFE.fullmatch(run_id or ''):
            raise LookupError('no such run')
        path = self.root / run_id
        if not (path / 'run.json').is_file():
            raise LookupError('no such run')
        return path

    def listing(self):
        rows = []
        for info_path in self.root.glob('*/run.json'):
            path = info_path.parent
            info = load(info_path, {})
            summary = load(path / 'summary.json')
            manifest = load(path / 'manifest.json')
            total = len(manifest['tasks']) if manifest else None
            done, passed = self.counts(path, info.get('benchmark'), summary)
            rows.append({'run_id': path.name, 'model_config': info.get('model_config'),
                         'suite_config': info.get('suite_config'), 'benchmark': info.get('benchmark'),
                         'oracle': bool(info.get('oracle')), 'regrade': info.get('regrade'),
                         'state': state(path, info, summary), 'done': done, 'total': total, 'passed': passed,
                         'started_at': info.get('started_at_unix'),
                         'elapsed_seconds': (summary or {}).get('elapsed_seconds')})
        return sorted(rows, key=lambda r: r['started_at'] or 0, reverse=True)

    def counts(self, path, benchmark, summary):
        rows = load(path / 'results.json', [])
        if summary is None and benchmark == 'tbv2':
            # Harbor rows are written at the end; count finished trials instead.
            trials = [load(p, {}) for p in (path / 'harbor' / 'job').glob('*/result.json')]
            finished = [t for t in trials if t.get('finished_at')]
            rewards = [((t.get('verifier_result') or {}).get('rewards') or {}).get('reward') for t in finished]
            return len(finished), sum(r == 1 for r in rewards)
        return sum(r['status'] != 'pending' for r in rows), sum(r['resolved'] is True for r in rows)

    def detail(self, run_id):
        path = self.path(run_id)
        info, summary = load(path / 'run.json', {}), load(path / 'summary.json')
        manifest = load(path / 'manifest.json') or {'tasks': []}
        benchmark = info.get('benchmark')
        saved = {r['task_id']: r for r in load(path / 'results.json', [])}
        now, running = time.time(), state(path, info, summary) == 'running'
        tasks = []
        for task in manifest['tasks']:
            task_id = task['id']
            result = saved.get(task_id) or {'task_id': task_id, 'status': 'pending', 'resolved': None}
            item = {k: result.get(k) for k in ROW_FIELDS}
            item['stage'] = None
            if running and result['status'] == 'pending':
                item.update(self.progress(path, benchmark, task_id, now))
            tasks.append(item)
        return {'run_id': run_id, 'info': info, 'state': state(path, info, summary), 'summary': summary,
                'comparison': load(path / 'comparison.json'), 'error': load(path / 'error.json'), 'tasks': tasks}

    def progress(self, path, benchmark, task_id, now):
        """Stage and live counters of a task that has no final row yet."""
        directory = task_directory(path, benchmark, task_id)
        if not directory.is_dir():
            return {'stage': 'queued'}
        if benchmark == 'swebench':
            stage = 'scoring' if (directory / 'scoring' / 'harness.log').exists() else 'agent'
            started = (directory / 'task.json').stat().st_mtime if (directory / 'task.json').exists() else None
        else:
            trial = load(directory / 'result.json', {})
            if trial.get('finished_at'):
                return {'stage': 'finished'}
            stage = next((name for key, name in (('verifier', 'verifier'), ('agent_execution', 'agent'),
                                                 ('agent_setup', 'agent_setup'),
                                                 ('environment_setup', 'environment_setup'))
                          if (trial.get(key) or {}).get('started_at')), 'environment_setup')
            started = directory.stat().st_mtime
        stream = event_file(directory)
        counters = live(events(stream)) if stream else {}
        return {'stage': stage, 'running_seconds': round(now - started) if started else None,
                **{k: v for k, v in counters.items() if v is not None}}

    def task(self, run_id, task_id):
        path = self.path(run_id)
        info = load(path / 'run.json', {})
        manifest = load(path / 'manifest.json') or {'tasks': []}
        if task_id not in [t['id'] for t in manifest['tasks']]:
            raise LookupError('no such task')
        benchmark = info.get('benchmark')
        result = next((r for r in load(path / 'results.json', []) if r['task_id'] == task_id), None)
        # A regrade keeps the agent artifacts in its source run.
        source = Path(info['regrade']['source']) if info.get('regrade') else path
        directory = task_directory(source, benchmark, task_id)
        stream = event_file(directory)
        return {'run_id': run_id, 'task_id': task_id, 'benchmark': benchmark, 'result': result,
                'timeline': timeline(events(stream)) if stream else [],
                'patch': None if info.get('oracle') else patch_text(directory, task_id),
                'failed_tests': failed_tests(task_directory(path, benchmark, task_id), task_id),
                'files': self.files(path, task_directory(path, benchmark, task_id))
                + (self.files(source, directory) if source != path else [])}

    def files(self, run, directory):
        if not directory.is_dir():
            return []
        found = []
        for p in sorted(directory.rglob('*')):
            if p.is_file() and not p.is_symlink() and allowed(p.name, run):
                found.append({'run_id': run.name, 'path': str(p.relative_to(run)), 'size': p.stat().st_size})
        return found

    def read(self, run_id, relative):
        """A file inside a registered run; never outside it, never through a symlink, never a gold patch."""
        run = self.path(run_id)
        parts = Path(relative or '').parts
        if not parts or Path(relative).is_absolute() or any(p in ('..', '.') for p in parts) or run.is_symlink():
            raise PermissionError('path not allowed')
        target = run
        for part in parts:
            target = target / part
            if target.is_symlink():
                raise PermissionError('path not allowed')
        if run.resolve() not in target.resolve().parents or not target.is_file() or not allowed(target.name, run):
            raise PermissionError('path not allowed')
        with target.open('rb') as stream:
            data = stream.read(FILE_LIMIT + 1)
        text = data[:FILE_LIMIT].decode(errors='replace')
        return redact(text) + ('\n[truncated]\n' if len(data) > FILE_LIMIT else '')


def allowed(name, run):
    if name in HIDDEN or name.endswith('.tmp'):
        return False
    # Oracle predictions are the gold patch.
    return not (name in ('preds.json', 'predictions.json') and load(run / 'run.json', {}).get('oracle'))


def clip(text):
    text = '' if text is None else text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    return text if len(text) <= TEXT_LIMIT else text[:TEXT_LIMIT] + f'\n[... {len(text) - TEXT_LIMIT} chars]'


def timeline(stream):
    items, start = [], stream[0]['timestamp'] if stream else 0
    for e in stream:
        kind, at = e['event'], round(e['timestamp'] - start, 1)
        if kind == 'llm_response':
            response = e.get('response') or {}
            native = (((response.get('extra') or {}).get('response') or {}).get('choices') or [{}])[0] or {}
            reasoning = response.get('reasoning_content') or (native.get('message') or {}).get('reasoning_content')
            items.append({'kind': 'model', 'at': at, 'index': e['response_index'], 'text': clip(response.get('content')),
                          'reasoning': clip(reasoning) if reasoning else None, 'usage': usage_of(e) or None})
        elif kind == 'tool_start':
            items.append({'kind': 'tool', 'at': at, 'index': e['response_index'], 'text': clip(e.get('command'))})
        elif kind == 'tool_end':
            result = e.get('result')
            output = result.get('output') if isinstance(result, dict) else result
            if output:
                items.append({'kind': 'output', 'at': at, 'index': e['response_index'], 'text': clip(output)})
        elif kind.endswith('_error') or kind == 'agent_end':
            items.append({'kind': 'error' if kind.endswith('_error') else 'end', 'at': at,
                          'index': e['response_index'], 'text': clip(' '.join(
                              str(e.get(k)) for k in ('exception_type', 'detail', 'exit_status') if e.get(k)))})
        elif kind == 'action_parse' and e.get('raw_error'):
            items.append({'kind': 'error', 'at': at, 'index': e['response_index'], 'text': clip(e['raw_error'])})
    return items


def patch_text(directory, task_id):
    candidate = directory / 'agent' / 'candidate.patch'
    preds = load(directory / 'agent' / 'preds.json') or {}
    submitted = (preds.get(task_id) or {}).get('model_patch')
    if submitted:
        return {'source': 'submitted', 'text': redact(clip(submitted))}
    text = candidate.read_text() if candidate.is_file() else ''
    # Not submitted: only graded under final-workspace.
    return {'source': 'final workspace (not submitted)', 'text': redact(clip(text))} if text.strip() else None


def failed_tests(directory, task_id):
    failed = []
    for path in directory.rglob('report.json') if directory.is_dir() else []:
        content = load(path, {})
        status = (content.get(task_id) or {}).get('tests_status') or {}
        for group, tests in status.items():
            failed += [f'{group}: {t}' for t in (tests or {}).get('failure', [])]
    for path in directory.rglob('ctrf.json') if directory.is_dir() else []:
        tests = ((load(path, {}).get('results') or {}).get('tests')) or []
        failed += [t.get('name', '?') for t in tests if t.get('status') not in ('passed', 'skipped')]
    return failed


def options(runs):
    """Suites and their models with recorded model runs, for the report form."""
    found = {}
    for r in runs.listing():
        if r['suite_config'] and r['model_config'] and not r['oracle'] and not r['regrade']:
            found.setdefault(r['suite_config'], set()).add(r['model_config'])
    return {suite: sorted(models) for suite, models in sorted(found.items())}


def comparison(runs, query):
    suite = (query.get('suite') or [None])[0]
    refs = [x for x in (query.get('models') or [''])[0].split(',') if x]
    benchmarks = tuple(query.get('benchmark') or ('swebench', 'tbv2'))
    if not suite or not refs:
        raise ValueError('choose a suite and at least one model')
    for ref in refs:
        if not SAFE.fullmatch(ref):
            raise ValueError('invalid model name')
    return build(refs, suite, benchmarks, runs.root)


PAGE_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self'; frame-src 'self'; base-uri 'none'; form-action 'none'")
REPORT_CSP = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'"


class Handler(BaseHTTPRequestHandler):
    runs = None
    hosts = ()

    def log_message(self, format, *args):
        pass

    def send(self, code, body, kind='application/json', csp=None):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header('Content-Type', kind + '; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', csp or "default-src 'none'; frame-ancestors 'self'")
        self.end_headers()
        self.wfile.write(data)

    def json(self, value, code=200):
        self.send(code, redact(json.dumps(value, ensure_ascii=False)))

    def do_GET(self):
        # Only answer to local host names: blocks DNS rebinding from a web page.
        if self.headers.get('Host', '') not in self.hosts:
            return self.send(403, 'forbidden host', 'text/plain')
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        one = lambda key: (query.get(key) or [''])[0]
        try:
            if url.path == '/':
                return self.send(200, PAGE, 'text/html', PAGE_CSP)
            if url.path == '/api/runs':
                return self.json(self.runs.listing())
            if url.path == '/api/run':
                return self.json(self.runs.detail(one('id')))
            if url.path == '/api/task':
                return self.json(self.runs.task(one('run'), one('task')))
            if url.path == '/api/file':
                return self.send(200, self.runs.read(one('run'), one('path')), 'text/plain')
            if url.path == '/api/options':
                return self.json(options(self.runs))
            if url.path == '/report':
                return self.send(200, redact(render(comparison(self.runs, query))), 'text/html', REPORT_CSP)
        except LookupError as exc:
            return self.json({'error': str(exc)}, 404)
        except PermissionError as exc:
            return self.json({'error': str(exc)}, 403)
        except (ValueError, KeyError, OSError) as exc:
            if url.path == '/report':
                return self.send(400, f'<!doctype html><meta charset="utf-8"><p>{html_escape(exc)}</p>',
                                 'text/html', REPORT_CSP)
            return self.json({'error': redact(f'{type(exc).__name__}: {exc}')}, 400)
        return self.json({'error': 'not found'}, 404)


def html_escape(exc):
    return html.escape(redact(f'{type(exc).__name__}: {exc}'))


def serve(runs_dir=Path('runs'), host='127.0.0.1', port=8765):
    handler = type('BoundHandler', (Handler,), {'runs': Runs(runs_dir)})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    port = server.server_address[1]  # The bound port when 0 was requested.
    handler.hosts = {f'{name}:{port}' for name in ('127.0.0.1', 'localhost', '[::1]', host)}
    return server


PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>swe-eval</title>
<style>body{font:14px/1.45 system-ui,sans-serif;margin:0;color:#1b1f24}
nav{background:#24292f;padding:8px 24px}nav a{color:#fff;margin-right:16px;text-decoration:none}
main{padding:16px 24px}table{border-collapse:collapse;margin:8px 0 16px}
th,td{border:1px solid #d0d7de;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f6f8fa}
td.num{text-align:right;font-variant-numeric:tabular-nums}.pass{background:#dafbe1}.fail{background:#ffebe9}
.running{background:#ddf4ff}.note{color:#57606a}.err{color:#cf222e}
pre{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;padding:8px;margin:4px 0;max-height:480px;overflow:auto}
.ev{border-left:3px solid #d0d7de;padding-left:8px;margin:8px 0}.ev.model{border-color:#0969da}
.ev.tool{border-color:#8250df}.ev.error{border-color:#cf222e}.ev.output{border-color:#d0d7de}
iframe{width:100%;height:80vh;border:1px solid #d0d7de}label{margin-right:12px}</style>
<nav><a href="#/">运行</a><a href="#/compare">对比</a></nav><main id="main"></main>
<script>
"use strict";
const main = document.getElementById("main");
let timer = null, token = 0;
function el(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") node.className = v; else if (k.startsWith("on")) node[k] = v; else node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined)
    node.append(kid instanceof Node ? kid : String(kid));
  return node;
}
function fmt(v) {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  if (typeof v === "boolean") return v ? "是" : "否";
  return String(v);
}
function td(v, cls) {
  return el("td", {class: (typeof v === "number" ? "num " : "") + (cls || "")}, v instanceof Node ? v : fmt(v));
}
function table(head, rows) {
  return el("table", {}, el("tr", {}, head.map(h => el("th", {}, h))), rows.map(r => el("tr", {}, r)));
}
function link(text, hash) { return el("a", {href: hash}, text); }
function q(params) { return new URLSearchParams(params).toString(); }
async function get(url) {
  const res = await fetch(url);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.status);
  return body;
}
function kind(r) { return r.resolved === true ? "pass" : r.resolved === false ? "fail" : r.stage ? "running" : ""; }
function when(t) { return t ? new Date(t * 1000).toLocaleString() : "–"; }
const STATES = {running: "运行中", complete: "完整", stopped: "已停止", incomplete: "不完整",
                dead: "进程已退出（无汇总）", unknown: "未知"};
async function runsPage() {
  const runs = await get("/api/runs");
  const rows = runs.map(r => [
    td(link(r.run_id, "#/run/" + encodeURIComponent(r.run_id))), td(r.model_config), td(r.suite_config),
    td(r.benchmark + (r.oracle ? " (oracle)" : r.regrade ? " (regrade)" : "")),
    td(STATES[r.state] || r.state, r.state === "running" ? "running" : ""),
    td((r.done ?? "–") + "/" + (r.total ?? "–")), td(r.passed), td(when(r.started_at)), td(r.elapsed_seconds)]);
  return [el("h1", {}, "运行"), runs.length ? table(["run", "模型", "套件", "benchmark", "状态", "进度", "通过",
          "开始", "耗时 s"], rows) : el("p", {class: "note"}, "runs/ 下还没有记录的 run。")];
}
async function runPage(id) {
  const r = await get("/api/run?" + q({id}));
  const s = r.summary || {};
  const rows = r.tasks.map(t => [
    td(link(t.task_id, "#/task/" + encodeURIComponent(id) + "/" + encodeURIComponent(t.task_id))),
    td(t.status, kind(t)), td(t.resolved), td(t.termination), td(t.stage), td(t.running_seconds ?? t.elapsed_seconds),
    td(t.agent_rounds), td(t.input_tokens), td(t.output_tokens), td(t.max_prompt_tokens), td(t.context_peak),
    td(t.scored_from), td(t.detail)]);
  const facts = [["状态", STATES[r.state] || r.state], ["模型", r.info.model_config], ["套件", r.info.suite_config],
    ["benchmark", r.info.benchmark], ["成功率", s.success_rate], ["通过", s.success_count],
    ["输入 token", s.input_tokens], ["输出 token", s.output_tokens], ["上下文峰值", s.context_peak],
    ["错误", r.error && r.error.error]];
  return [el("h1", {}, id), table(["项", "值"], facts.map(([k, v]) => [td(k), td(v)])),
          el("h2", {}, "逐题"), table(["任务", "状态", "通过", "终止", "阶段", "耗时 s", "轮数", "输入 token",
          "输出 token", "最大 prompt", "上下文峰值", "评分来源", "说明"], rows),
          r.state === "running" ? el("p", {class: "note"}, "运行中：每 3 秒刷新；最终指标在 run 结束时写入。") : null];
}
function timelineItem(e) {
  const title = {model: "模型", tool: "工具调用", output: "工具输出", error: "错误", end: "结束"}[e.kind];
  const box = el("div", {class: "ev " + e.kind}, el("div", {class: "note"},
    "#" + e.index + " · " + title + " · +" + e.at + "s" + (e.usage ? " · prompt " + fmt(e.usage.prompt_tokens)
    + " / completion " + fmt(e.usage.completion_tokens) : "")));
  if (e.reasoning) box.append(el("details", {}, el("summary", {}, "reasoning"), el("pre", {}, e.reasoning)));
  if (e.kind === "output") box.append(el("details", {}, el("summary", {}, "输出"), el("pre", {}, e.text)));
  else box.append(el("pre", {}, e.text));
  return box;
}
async function taskPage(run, task) {
  const t = await get("/api/task?" + q({run, task}));
  const r = t.result || {};
  const parts = [el("h1", {}, task), el("p", {}, link("← " + run, "#/run/" + encodeURIComponent(run))),
    table(["状态", "通过", "终止", "评分来源", "说明"], [[td(r.status, kind(r)), td(r.resolved), td(r.termination),
    td(r.scored_from), td(r.detail)]])];
  if (t.failed_tests.length) parts.push(el("h2", {}, "失败测试"), el("pre", {}, t.failed_tests.join("\\n")));
  parts.push(el("h2", {}, "补丁"), t.patch ? [el("p", {class: "note"}, "来源：" + t.patch.source),
             el("pre", {}, t.patch.text)] : el("p", {class: "note"}, "无补丁"));
  parts.push(el("h2", {}, "时间线"), t.timeline.length ? t.timeline.map(timelineItem)
             : el("p", {class: "note"}, "没有 agent 事件。"));
  parts.push(el("h2", {}, "原始文件"), el("ul", {}, t.files.map(f => el("li", {}, el("a", {
    href: "/api/file?" + q({run: f.run_id, path: f.path}), target: "_blank", rel: "noopener"},
    f.run_id === run ? f.path : f.run_id + "/" + f.path), " (" + f.size + " B)"))));
  return parts;
}
async function comparePage() {
  const options = await get("/api/options");
  const suites = Object.keys(options);
  if (!suites.length) return [el("h1", {}, "对比"), el("p", {class: "note"}, "还没有可对比的模型 run。")];
  const picked = [];
  const frame = el("iframe", {title: "report", sandbox: ""});
  const models = el("div", {});
  const order = el("p", {class: "note"});
  const suite = el("select", {onchange: () => fill()}, suites.map(s => el("option", {value: s}, s)));
  const benches = ["swebench", "tbv2"].map(b => el("input", {type: "checkbox", value: b, checked: ""}));
  function fill() {
    picked.length = 0;
    models.replaceChildren(...options[suite.value].map(m => el("label", {}, el("input", {type: "checkbox", value: m,
      onchange: e => { if (e.target.checked) picked.push(m); else picked.splice(picked.indexOf(m), 1);
                       order.textContent = "baseline 在前：" + picked.join(" → "); }}), " " + m)));
    order.textContent = "按勾选顺序排列，第一个是 baseline。";
  }
  fill();
  const go = el("button", {onclick: () => {
    const params = new URLSearchParams({suite: suite.value, models: picked.join(",")});
    benches.filter(b => b.checked).forEach(b => params.append("benchmark", b.value));
    frame.src = "/report?" + params;
  }}, "生成报告");
  return [el("h1", {}, "对比"), el("p", {}, "套件 ", suite), models, order,
          el("p", {}, benches.map(b => el("label", {}, b, " " + b.value))), go, frame];
}
async function route() {
  clearTimeout(timer);
  const mine = ++token;
  const parts = location.hash.replace(/^#\\/?/, "").split("/").map(decodeURIComponent);
  let page, poll = true;
  if (parts[0] === "run") page = () => runPage(parts[1]);
  else if (parts[0] === "task") page = () => taskPage(parts[1], parts[2]);
  else if (parts[0] === "compare") { page = comparePage; poll = false; }
  else page = runsPage;
  async function draw() {
    let content;
    try { content = await page(); } catch (err) { content = [el("p", {class: "err"}, String(err.message || err))]; }
    if (mine !== token) return;
    // Keep open <details> while polling.
    const open = [...main.querySelectorAll("details")].map(d => d.open);
    const y = window.scrollY;
    main.replaceChildren(...[content].flat());
    main.querySelectorAll("details").forEach((d, i) => { if (open[i]) d.open = true; });
    window.scrollTo(0, y);
    if (poll) timer = setTimeout(draw, 3000);
  }
  await draw();
}
window.addEventListener("hashchange", route);
route();
</script></html>
'''
