import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from swe_eval.config import digest
from swe_eval.io import read_json, write_json
from swe_eval.results import row
from swe_eval.runner import resume_run
from tests.test_framework import make_config


def manifest():
    tasks = [{'id': f't{i}', 'record': {'instance_id': f't{i}', 'patch': 'gold', 'problem_statement': 'x'}}
             for i in range(4)]
    return {'schema_version': 2, 'benchmark': 'swebench', 'name': 'small', 'tasks': tasks,
            'tasks_sha256': digest(tasks)}


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        lock = self.root / 'environments' / 'swebench' / 'uv.lock'
        lock.parent.mkdir(parents=True)
        lock.write_text('locked')
        self.output = self.root / 'runs' / 'crashed'
        self.output.mkdir(parents=True)
        # A run whose process died mid-way: no summary.json, t1 in flight, t3 never started.
        write_json(self.output / 'run.json', {'benchmark': 'swebench', 'manifest': 'small', 'run_id': 'crashed',
                                              'oracle': False, 'started_at_unix': 1.0, 'pid': None})
        write_json(self.output / 'config.json', make_config(suite={'concurrency': 4}).snapshot())
        write_json(self.output / 'manifest.json', manifest())
        write_json(self.output / 'comparison.json', {
            'benchmark': 'swebench', 'tasks_sha256': manifest()['tasks_sha256'], 'concurrency': 4,
            'agent_sources_sha256': 'source', 'installed_distributions_sha256': digest([]),
            'environment_lock_sha256': hashlib.sha256(b'locked').hexdigest(), 'runner_sha256': 'old'})
        done = dict(row('t0', 'success'), resolved=True)
        failed = dict(row('t2', 'api_error', 'API timeout'))
        write_json(self.output / 'results.json', [done, row('t1'), failed, row('t3')])
        (self.output / 'tasks' / 't0' / 'agent').mkdir(parents=True)
        (self.output / 'tasks' / 't1' / 'agent').mkdir(parents=True)
        (self.output / 'tasks' / 't1' / 'agent' / 'partial.txt').write_text('half done')
        self.started = []

    def resume(self, sources='source', preflight=True, **kwargs):
        def fake_worker(benchmark, args, log, **_):
            if args[0] == 'metadata':
                write_json(args[2], {'versions': {}, 'all_distributions': [], 'python': 'test',
                                     'agent_sources_sha256': sources})

        def fake_execute(command, cwd, log, env=None, timeout=None, grace=30):
            target = Path(command[-1])
            task_id = target.parent.name
            self.started.append(task_id)
            self.assertFalse((target / 'partial.txt').exists(), 'partial artifacts must be removed')
            write_json(target / task_id / f'{task_id}.traj.json', {'info': {'exit_status': 'LimitsExceeded'}})
            return {'elapsed_seconds': 1, 'returncode': 0, 'timed_out': False}

        checks = {'ok': preflight, 'checks': {'api': {'ok': preflight}}}
        with patch.dict(os.environ, {'EVAL_TEST_KEY': 'resume-test-key'}), patch('swe_eval.runner.ROOT', self.root), \
                patch('swe_eval.runner.worker', side_effect=fake_worker), \
                patch('swe_eval.runner.doctor', return_value=checks), \
                patch('swe_eval.runner.environment_python', return_value=sys.executable), \
                patch('swe_eval.runner.execute', side_effect=fake_execute), \
                patch('swe_eval.runner.docker_oom_events', return_value=None), \
                patch('swe_eval.runner.cleanup_containers'), patch('sys.stdout', io.StringIO()):
            return resume_run(self.output, **kwargs)

    def test_resume_keeps_finished_tasks_and_reruns_unfinished(self):
        output, summary = self.resume(concurrency=2)
        self.assertEqual(sorted(self.started), ['t1', 't3'])
        rows = read_json(output / 'results.json')
        self.assertEqual([r['status'] for r in rows], ['success', 'budget_exhausted', 'api_error', 'budget_exhausted'])
        self.assertTrue(rows[0]['resolved'])
        # The kept api_error still makes the run incomplete.
        self.assertFalse(summary['complete'])
        self.assertIsNone(summary['elapsed_seconds'], 'the crashed segment has no recorded time')
        info = read_json(output / 'run.json')
        self.assertEqual(len(info['resumes']), 1)
        self.assertEqual(info['resumes'][0]['rerun'], ['t1', 't3'])
        self.assertEqual(info['resumes'][0]['concurrency'], 2)
        comparison = read_json(output / 'comparison.json')
        self.assertEqual(comparison['concurrency'], 4)
        self.assertEqual(comparison['resumes'][0]['concurrency'], 2)

    def test_retry_errors_reruns_api_errors_and_completes(self):
        _, summary = self.resume(retry_errors=True)
        self.assertEqual(sorted(self.started), ['t1', 't2', 't3'])
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['success_count'], 1)
        with self.assertRaisesRegex(ValueError, 'no unfinished tasks'):
            self.resume()

    def test_resume_after_a_summary_adds_elapsed_time(self):
        write_json(self.output / 'summary.json', {'elapsed_seconds': 100.0, 'stopped': True})
        _, summary = self.resume()
        self.assertGreaterEqual(summary['elapsed_seconds'], 100.0)

    def test_refuses_changed_agent_sources_and_leaves_run_untouched(self):
        before = (self.output / 'results.json').read_text()
        with self.assertRaisesRegex(ValueError, 'agent_sources_sha256 changed'):
            self.resume(sources='edited')
        self.assertEqual((self.output / 'results.json').read_text(), before)
        self.assertTrue((self.output / 'tasks' / 't1' / 'agent' / 'partial.txt').exists())
        self.assertEqual(self.started, [])

    def test_failed_preflight_leaves_run_resumable(self):
        before = (self.output / 'results.json').read_text()
        with self.assertRaisesRegex(RuntimeError, 'Preflight failed'):
            self.resume(preflight=False)
        self.assertEqual((self.output / 'results.json').read_text(), before)
        self.assertNotIn('resumes', read_json(self.output / 'run.json'))
        self.assertTrue((self.output / 'doctor.resume.json').exists())

    def test_refuses_live_runs_and_other_benchmarks(self):
        info = read_json(self.output / 'run.json')
        with patch('swe_eval.runner.alive', return_value=True):
            write_json(self.output / 'run.json', dict(info, pid=999999))
            with self.assertRaisesRegex(ValueError, 'still running'):
                self.resume()
        write_json(self.output / 'run.json', dict(info, benchmark='tbv2'))
        with self.assertRaisesRegex(ValueError, 'SWE-bench'):
            self.resume()


if __name__ == '__main__':
    unittest.main()
