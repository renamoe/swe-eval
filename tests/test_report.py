import io
from html.parser import HTMLParser
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from swe_eval.__main__ import main
from swe_eval.config import digest
from swe_eval.io import read_json, write_json
from swe_eval.report import report
from swe_eval.results import row, summarize
from swe_eval.runner import regrade

IDS = ['a__a-1', 'b__b-1', 'c__c-1']


def settings(**suite):
    return {'model': {'context_length': 131072, 'sampling': {'temperature': 0.6}},
            'suite': {'max_turns': 50, 'swe_scoring': 'submitted-only', **suite}}


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runs = Path(self.tmp.name) / 'runs'

    def make_run(self, name, resolved, model=None, benchmark='swebench', started=1.0, complete=True,
                 suite_settings=None, agent='src', tasks_digest=None):
        folder = self.runs / name
        folder.mkdir(parents=True)
        tasks = [{'id': i, 'record': {'instance_id': i, 'patch': 'GOLD-PATCH-' + i}} for i in IDS]
        write_json(folder / 'manifest.json', {'schema_version': 2, 'benchmark': benchmark, 'name': 'small',
                                              'tasks': tasks, 'tasks_sha256': tasks_digest or digest(tasks)})
        write_json(folder / 'comparison.json', {'benchmark': benchmark, 'tasks_sha256': digest(tasks),
                                                'settings': suite_settings or settings(),
                                                'agent_sources_sha256': agent})
        write_json(folder / 'run.json', {'benchmark': benchmark, 'manifest': 'small', 'run_id': name,
                                         'model_config': model or name, 'suite_config': 'small-v1',
                                         'oracle': False, 'started_at_unix': started})
        write_json(folder / 'config.json', {'model_config': model or name, 'model': {'checkpoint': name}})
        rows = []
        for task_id, value in zip(IDS, resolved):
            result = row(task_id, 'pending' if value is None else 'success' if value else 'task_failed')
            result.update(resolved=value, termination='submitted' if value is not None else None)
            if value is None and not complete:
                result['status'] = 'stopped'
            rows.append(result)
        summarize(folder, rows, 10, len(IDS))
        return folder

    def test_matrix_gains_regressions_and_rate_delta(self):
        base = self.make_run('base', [True, False, False])
        ckpt = self.make_run('ckpt', [False, True, True])
        result = report([base, ckpt], self.runs / 'r')
        part = result['benchmarks'][0]
        self.assertEqual(part['differences'], [])
        first, second = part['models']
        self.assertEqual((second['gained'], second['regressed'], second['passed_count']), (2, 1, 2))
        self.assertAlmostEqual(second['success_rate_delta'], 1 / 3, places=5)
        self.assertEqual(first['success_rate_delta'], 0)
        self.assertEqual([c['change'] for c in part['matrix'][0]['runs']], [None, 'regressed'])
        rows = (self.runs / 'r' / 'report.csv').read_text().splitlines()
        self.assertEqual(len(rows), 1 + 2 * len(IDS))

    def test_incompatible_runs_are_rejected(self):
        base = self.make_run('base', [True, False, False])
        cases = [('scoring', {'suite_settings': settings(swe_scoring='final-workspace')}, 'swe_scoring'),
                 ('agent', {'agent': 'other'}, 'agent_sources_sha256')]
        for name, kwargs, message in cases:
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                report([base, self.make_run(name, [True] * 3, **kwargs)], self.runs / ('r-' + name))
        other = self.make_run('tasks', [True] * 3)
        manifest = read_json(other / 'manifest.json')
        manifest['tasks'] = manifest['tasks'][:2]
        manifest['tasks_sha256'] = digest(manifest['tasks'])
        write_json(other / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'cover the frozen task list'):
            report([base, other], self.runs / 'r-tasks')
        tampered = self.make_run('tampered', [True] * 3, tasks_digest='0' * 64)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            report([base, tampered], self.runs / 'r-tampered')
        self.assertFalse((self.runs / 'r-scoring').exists())

    def test_other_differences_are_listed_not_blocking(self):
        base = self.make_run('base', [True, False, False])
        ckpt = self.make_run('ckpt', [True, True, False], suite_settings=settings(max_turns=100))
        part = report([base, ckpt], self.runs / 'r')['benchmarks'][0]
        self.assertEqual(part['differences'], [{'key': 'settings.suite.max_turns', 'values': [50, 100]}])
        self.assertIn('settings.suite.max_turns', (self.runs / 'r' / 'report.html').read_text())

    def test_incomplete_run_has_no_rate_delta_and_pairs_scored_tasks(self):
        base = self.make_run('base', [True, False, False])
        stopped = self.make_run('stopped', [False, True, None], complete=False)
        model = report([base, stopped], self.runs / 'r')['benchmarks'][0]['models'][1]
        self.assertFalse(model['complete'])
        self.assertIsNone(model['success_rate'])
        self.assertIsNone(model['success_rate_delta'])
        self.assertEqual((model['paired_count'], model['scored_count'], model['gained'], model['regressed']),
                         (2, 2, 1, 1))
        html = (self.runs / 'r' / 'report.html').read_text()
        self.assertIn('不给成功率差值', html)

    def test_html_is_offline_and_never_exports_gold_patches(self):
        base = self.make_run('base<x>', [True, False, False])
        report([base, self.make_run('ckpt', [True, True, False])], self.runs / 'r')
        text = (self.runs / 'r' / 'report.html').read_text()
        parser = Tags()
        parser.feed(text)
        tags = {t for t, _ in parser.tags}
        self.assertIn('style', tags)
        self.assertFalse(tags & {'script', 'link', 'img', 'iframe'})
        self.assertFalse(any(v and v.startswith(('http:', 'https:', '//')) for _, a in parser.tags for v in a.values()))
        self.assertIn('base&lt;x&gt;', text)
        for name in ('report.html', 'report.json', 'report.csv'):
            self.assertNotIn('GOLD-PATCH', (self.runs / 'r' / name).read_text())

    def test_models_resolve_to_latest_complete_run_per_benchmark(self):
        self.make_run('base-old', [False] * 3, model='base', started=1)
        self.make_run('base-new', [True, False, False], model='base', started=2)
        self.make_run('base-stopped', [True, None, None], model='base', started=3, complete=False)
        self.make_run('ckpt-1', [True] * 3, model='ckpt', started=2)
        self.make_run('ckpt-tb', [True] * 3, model='ckpt', benchmark='tbv2', started=2)
        self.make_run('base-tb', [False] * 3, model='base', benchmark='tbv2', started=2)
        result = report(['base', 'ckpt'], self.runs / 'r', 'small-v1', ('swebench', 'tbv2'), self.runs)
        self.assertEqual([[m['run_id'] for m in b['models']] for b in result['benchmarks']],
                         [['base-new', 'ckpt-1'], ['base-tb', 'ckpt-tb']])
        with self.assertRaisesRegex(ValueError, 'No swebench run of model missing'):
            report(['base', 'missing'], self.runs / 'r2', 'small-v1', ('swebench',), self.runs)

    def test_compare_alias_writes_report(self):
        base = self.make_run('base', [True, False, False])
        ckpt = self.make_run('ckpt', [True, True, False])
        with patch('sys.stdout', io.StringIO()) as out:
            self.assertEqual(main(['compare', str(base), str(ckpt)]), 0)
        self.assertIn('report.html', out.getvalue())
        self.assertTrue((self.runs / 'compare-base-vs-ckpt' / 'report.json').exists())
        tb = self.make_run('tb', [True] * 3, benchmark='tbv2')
        with patch('sys.stderr', io.StringIO()) as err:
            self.assertEqual(main(['compare', str(base), str(tb), '--output', str(self.runs / 'x')]), 2)
        self.assertIn('same benchmark', err.getvalue())


class RegradeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'runs' / 'src'
        tasks = [{'id': i, 'record': {'instance_id': i, 'patch': 'gold'}} for i in IDS]
        write_json(self.source / 'manifest.json', {'schema_version': 2, 'benchmark': 'swebench', 'name': 'small',
                                                   'tasks': tasks, 'tasks_sha256': digest(tasks)})
        write_json(self.source / 'comparison.json', {'benchmark': 'swebench', 'tasks_sha256': digest(tasks),
                                                     'settings': settings(), 'agent_sources_sha256': 'src'})
        write_json(self.source / 'run.json', {'benchmark': 'swebench', 'run_id': 'src', 'model_config': 'm',
                                              'suite_config': 'small-v1', 'oracle': False, 'started_at_unix': 1})
        write_json(self.source / 'config.json', {'model_config': 'm', 'model': {},
                                                 'suite': {'swe_scoring': 'submitted-only', 'swe_score_timeout': 60}})
        rows = [row(IDS[0], 'success'), row(IDS[1], 'budget_exhausted', 'Agent exit: LimitsExceeded'),
                row(IDS[2], 'budget_exhausted', 'Agent exit: LimitsExceeded')]
        rows[0]['resolved'], rows[1]['resolved'], rows[2]['resolved'] = True, False, False
        for task_id, patch_text in ((IDS[1], 'diff --git a/x b/x'), (IDS[2], '')):
            agent = self.source / 'tasks' / task_id / 'agent'
            agent.mkdir(parents=True)
            (agent / 'candidate.patch').write_text(patch_text)
            write_json(self.source / 'tasks' / task_id / 'dataset.json', [tasks[IDS.index(task_id)]['record']])
        summarize(self.source, rows, 10, len(IDS))

    def fake_execute(self, resolved=True, calls=None):
        def execute(command, cwd, log, env=None, timeout=None, grace=30):
            (calls if calls is not None else []).append(command)
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            Path(log).write_text('ok')
            write_json(Path(cwd) / 'logs' / 'report.json', {command[command.index('--instance_ids') + 1]:
                                                            {'resolved': resolved}})
            return {'elapsed_seconds': 1, 'returncode': 0, 'timed_out': False}
        return execute

    def snapshot(self):
        return {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob('*') if p.is_file()}

    def test_regrade_writes_side_run_and_leaves_source_unchanged(self):
        before, calls = self.snapshot(), []
        with patch('swe_eval.runner.execute', side_effect=self.fake_execute(calls=calls)),                 patch('swe_eval.runner.environment_python', return_value=sys.executable),                 patch('swe_eval.runner.cleanup_containers') as cleanup, patch('sys.stdout', io.StringIO()):
            output, summary = regrade(self.source)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(output.name, 'src__final-workspace')
        cleanup.assert_called_once_with(output, output.name)
        # Only the non-empty candidate of an exhausted budget is graded.
        self.assertEqual(len(calls), 1)
        self.assertEqual(read_json(output / 'tasks' / IDS[1] / 'predictions.json')[IDS[1]]['model_patch'],
                         'diff --git a/x b/x')
        rows = {r['task_id']: r for r in read_json(output / 'results.json')}
        self.assertEqual((rows[IDS[1]]['scored_from'], rows[IDS[1]]['resolved'], rows[IDS[1]]['status']),
                         ('candidate', True, 'budget_exhausted'))
        self.assertIsNone(rows[IDS[2]]['scored_from'])
        self.assertEqual((summary['success_count'], summary['regraded_count']), (2, 1))
        self.assertTrue(summary['complete'])
        self.assertEqual(read_json(output / 'comparison.json')['settings']['suite']['swe_scoring'], 'final-workspace')
        self.assertEqual(read_json(output / 'run.json')['regrade']['policy'], 'final-workspace')
        # Different scoring policies never form a controlled comparison.
        with self.assertRaisesRegex(ValueError, 'swe_scoring'):
            report([self.source, output], self.source.parent / 'r')
        with self.assertRaises(FileExistsError), patch('sys.stdout', io.StringIO()):
            regrade(self.source)

    def test_regrade_rejects_wrong_inputs(self):
        config = read_json(self.source / 'config.json')
        config['suite']['swe_scoring'] = 'final-workspace'
        write_json(self.source / 'config.json', config)
        with self.assertRaisesRegex(ValueError, 'already scored'):
            regrade(self.source)
        info = read_json(self.source / 'run.json')
        write_json(self.source / 'run.json', dict(info, benchmark='tbv2'))
        with self.assertRaisesRegex(ValueError, 'SWE-bench model runs only'):
            regrade(self.source)
        with self.assertRaisesRegex(ValueError, 'only'):
            regrade(self.source, policy='submitted-only')


if __name__ == '__main__':
    unittest.main()
