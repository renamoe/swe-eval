import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from swe_eval.config import digest
from swe_eval.io import write_json
from swe_eval.results import row, summarize
from swe_eval.web import Runs, serve

IDS = ['a__a-1', 'b__b-1']


def make_run(root, name, rows=None, finished=True, oracle=False, pid=None):
    folder = root / name
    tasks = [{'id': i, 'record': {'instance_id': i, 'patch': 'GOLD-PATCH-' + i}} for i in IDS]
    write_json(folder / 'manifest.json', {'schema_version': 2, 'benchmark': 'swebench', 'name': 'small',
                                          'tasks': tasks, 'tasks_sha256': digest(tasks)})
    write_json(folder / 'comparison.json', {'benchmark': 'swebench', 'tasks_sha256': digest(tasks),
                                            'settings': {'suite': {'swe_scoring': 'submitted-only'}},
                                            'agent_sources_sha256': 'src'})
    write_json(folder / 'run.json', {'benchmark': 'swebench', 'manifest': 'small', 'run_id': name,
                                     'model_config': 'm', 'suite_config': 'small-v1', 'oracle': oracle,
                                     'started_at_unix': 1, 'pid': pid})
    rows = rows or [row(i) for i in IDS]
    if finished:
        summarize(folder, rows, 1, len(IDS))
    else:
        write_json(folder / 'results.json', rows)
    return folder


class WebTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'runs'
        done = row(IDS[0], 'success')
        done['resolved'] = True
        self.live = make_run(self.root, 'live', [done, row(IDS[1])], finished=False, pid=os.getpid())
        task = self.live / 'tasks' / IDS[1]
        write_json(task / 'task.json', {'instance_id': IDS[1], 'patch': 'GOLD-PATCH-x'})
        (task / 'agent' / IDS[1]).mkdir(parents=True)
        events = [{'event': 'llm_start', 'timestamp': 1, 'response_index': 1},
                  {'event': 'llm_response', 'timestamp': 2, 'response_index': 1,
                   'response': {'content': 'ls', 'extra': {'response': {
                       'usage': {'prompt_tokens': 100, 'completion_tokens': 5},
                       'choices': [{'message': {'reasoning_content': 'think'}}]}}}},
                  {'event': 'tool_start', 'timestamp': 3, 'response_index': 1, 'command': 'ls'}]
        (task / 'agent' / IDS[1] / 'events.jsonl').write_text(
            ''.join(json.dumps(e) + '\n' for e in events) + '{"event": "llm_st')  # Line being written.
        (task / 'agent' / 'candidate.patch').write_text('diff --git a/x b/x')
        self.runs = Runs(self.root)

    def test_running_run_shows_progress_and_live_counters(self):
        with patch('swe_eval.web.alive', return_value=True):
            listing = {r['run_id']: r for r in self.runs.listing()}
            detail = self.runs.detail('live')
        self.assertEqual((listing['live']['state'], listing['live']['done'], listing['live']['total'],
                          listing['live']['passed']), ('running', 1, 2, 1))
        current = detail['tasks'][1]
        self.assertEqual((current['stage'], current['agent_rounds'], current['input_tokens'],
                          current['max_prompt_tokens']), ('agent', 1, 100, 100))
        task = self.runs.task('live', IDS[1])
        self.assertEqual([e['kind'] for e in task['timeline']], ['model', 'tool'])
        self.assertEqual(task['timeline'][0]['reasoning'], 'think')
        self.assertEqual(task['patch']['source'], 'final workspace (not submitted)')

    def test_tb_stage_without_result_json_is_inferred_from_files(self):
        trial = self.root / 'tb' / 'harbor' / 'job' / 'regex-log__abc'
        trial.mkdir(parents=True)
        stage = lambda: self.runs.progress(self.root / 'tb', 'tbv2', 'regex-log', 0)['stage']
        with patch('swe_eval.web.task_directory', return_value=trial):
            self.assertEqual(stage(), 'environment_setup')
            write_json(trial / 'agent' / 'trajectory.json', {'steps': []})
            self.assertEqual(stage(), 'agent')
            write_json(trial / 'verifier' / 'test-stdout.txt', {})
            self.assertEqual(stage(), 'verifier')

    def test_finished_tb_trial_shows_verdict_before_job_ends(self):
        folder = self.root / 'tb'
        write_json(folder / 'manifest.json', {'tasks': [{'id': 'regex-log'}, {'id': 'raman-fitting'}]})
        write_json(folder / 'run.json', {'benchmark': 'tbv2', 'pid': os.getpid()})
        write_json(folder / 'results.json', [row('regex-log'), row('raman-fitting')])
        write_json(folder / 'harbor' / 'job' / 'regex-log__abc' / 'result.json', {
            'task_name': 'regex-log', 'started_at': '2026-01-01T00:00:00', 'finished_at': '2026-01-01T00:01:40',
            'verifier_result': {'rewards': {'reward': 1.0}}, 'agent_result': {}})
        (folder / 'harbor' / 'job' / 'raman-fitting__def').mkdir(parents=True)
        with patch('swe_eval.web.alive', return_value=True):
            tasks = {t['task_id']: t for t in self.runs.detail('tb')['tasks']}
        self.assertEqual((tasks['regex-log']['status'], tasks['regex-log']['resolved'],
                          tasks['regex-log']['elapsed_seconds']), ('success', True, 100))
        self.assertEqual((tasks['raman-fitting']['status'], tasks['raman-fitting']['stage']),
                         ('pending', 'environment_setup'))

    def test_run_without_summary_and_process_is_dead(self):
        with patch('swe_eval.web.alive', return_value=False):
            self.assertEqual(self.runs.detail('live')['state'], 'dead')
        make_run(self.root, 'done')
        self.assertEqual(self.runs.detail('done')['state'], 'incomplete')

    def test_file_reads_stay_inside_registered_runs(self):
        good = f'tasks/{IDS[1]}/agent/candidate.patch'
        self.assertIn('diff --git', self.runs.read('live', good))
        (self.root / 'secret.txt').write_text('SECRET')
        (self.live / 'link').symlink_to(self.root / 'secret.txt')
        (self.live / 'dirlink').symlink_to(self.root)
        for path in ('../secret.txt', '/etc/passwd', 'link', 'dirlink/secret.txt', 'manifest.json',
                     f'tasks/{IDS[1]}/task.json', f'tasks/{IDS[1]}/../../../secret.txt', '', '.'):
            with self.subTest(path), self.assertRaises(PermissionError):
                self.runs.read('live', path)
        for run in ('..', '', 'missing', '../runs', 'live/..'):
            with self.subTest(run), self.assertRaises(LookupError):
                self.runs.read(run, 'run.json')

    def test_oracle_gold_patch_is_never_exposed(self):
        oracle = make_run(self.root, 'oracle', oracle=True)
        agent = oracle / 'tasks' / IDS[0] / 'agent'
        write_json(agent / 'preds.json', {IDS[0]: {'model_patch': 'GOLD-PATCH-a'}})
        write_json(oracle / 'tasks' / IDS[0] / 'predictions.json', {IDS[0]: {'model_patch': 'GOLD-PATCH-a'}})
        task = self.runs.task('oracle', IDS[0])
        self.assertIsNone(task['patch'])
        self.assertEqual(task['files'], [])
        with self.assertRaises(PermissionError):
            self.runs.read('oracle', f'tasks/{IDS[0]}/agent/preds.json')

    def test_http_server_is_local_read_only_and_never_serves_gold(self):
        server = serve(self.root, '127.0.0.1', 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_address[1]}'

        def get(path, **headers):
            try:
                with urlopen(Request(base + path, headers=headers), timeout=10) as response:
                    return response.status, response.headers, response.read().decode()
            except HTTPError as exc:
                return exc.code, exc.headers, exc.read().decode()

        status, headers, page = get('/')
        self.assertEqual(status, 200)
        self.assertIn('connect-src', headers['Content-Security-Policy'])
        self.assertNotIn('http', page.split('<script>')[1].replace('http.server', ''))  # No external fetches.
        self.assertEqual(get('/api/runs')[0], 200)
        self.assertEqual(get('/api/runs', Host='evil.example:80')[0], 403)
        for query in ({'run': 'live', 'path': '../secret'}, {'run': 'live', 'path': 'manifest.json'},
                      {'run': '..', 'path': 'run.json'}):
            self.assertIn(get('/api/file?' + urlencode(query))[0], (403, 404))
        status, headers, text = get('/api/file?' + urlencode({'run': 'live', 'path': 'run.json'}))
        self.assertEqual((status, headers['Content-Type']), (200, 'text/plain; charset=utf-8'))
        make_run(self.root, 'base', [dict(row(i, 'success'), resolved=True) for i in IDS])
        status, headers, text = get('/report?' + urlencode({'suite': 'small-v1', 'models': 'm',
                                                              'benchmark': 'swebench'}))
        self.assertEqual(status, 200)
        self.assertNotIn('script-src', headers['Content-Security-Policy'])
        self.assertEqual(get('/report?suite=small-v1&models=../x')[0], 400)
        for path in ('/api/runs', '/api/run?id=live', f'/api/task?run=live&task={IDS[1]}', '/api/options'):
            self.assertNotIn('GOLD-PATCH', get(path)[2])
        try:
            with urlopen(Request(base + '/api/runs', method='POST', data=b'x'), timeout=10):
                pass
            self.fail('POST accepted')
        except HTTPError as exc:
            self.assertEqual(exc.code, 501)


if __name__ == '__main__':
    unittest.main()
