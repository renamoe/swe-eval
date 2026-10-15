import io
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from swe_eval import io as swe_io
from swe_eval.io import read_json, write_json
from swe_eval.results import row
from swe_eval.runner import run_swe, tb_config
from tests.test_framework import make_config
from tests.test_interrupt import TREE, alive


def tasks(count):
    return [{'id': f't{i}', 'record': {'instance_id': f't{i}', 'patch': 'gold', 'problem_statement': 'x'}}
            for i in range(count)]


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / 'run'
        self.output.mkdir()
        environment = patch.dict(os.environ, {'EVAL_TEST_KEY': 'concurrency-test-key'})
        environment.start()
        self.addCleanup(environment.stop)

    def run_swe(self, config, items, fake_execute):
        rows = [row(t['id']) for t in items]
        with patch('swe_eval.runner.worker'), patch('swe_eval.runner.environment_python', return_value=sys.executable), \
                patch('swe_eval.runner.execute', side_effect=fake_execute), patch('sys.stdout', io.StringIO()):
            run_swe(config, items, self.output, False, rows)
        return rows

    def test_two_tasks_overlap_and_results_keep_manifest_order(self):
        spans, lock = {}, threading.Lock()
        def fake_execute(command, cwd, log, env=None, timeout=None, grace=30):
            target = Path(command[-1])
            task_id = target.parent.name
            begin = time.monotonic()
            Path(log).write_text('Traceback: worker crashed\n')
            time.sleep(0.4 if task_id == 't0' else 0.2)
            write_json(target / task_id / f'{task_id}.traj.json', {'info': {'exit_status': 'LimitsExceeded'}})
            with lock:
                spans[task_id] = (begin, time.monotonic())
            # t1 fails as an environment error; the other tasks must still finish.
            return {'elapsed_seconds': 0.2, 'returncode': 1 if task_id == 't1' else 0, 'timed_out': False}
        started = time.monotonic()
        rows = self.run_swe(make_config(suite={'concurrency': 2}), tasks(4), fake_execute)
        elapsed = time.monotonic() - started
        self.assertEqual([r['task_id'] for r in rows], ['t0', 't1', 't2', 't3'])
        self.assertEqual(read_json(self.output / 'results.json'), rows)
        self.assertNotIn('pending', [r['status'] for r in rows])
        self.assertEqual(rows[0]['status'], 'budget_exhausted')
        self.assertLess(spans['t1'][0], spans['t0'][1], 't0 and t1 must run at the same time')
        self.assertLess(elapsed, 0.75)  # Serial would take 1.0s.

    def test_interrupt_kills_all_running_tasks_at_once_and_skips_scoring(self):
        commands, pids = [], []
        def fake_execute(command, cwd, log, env=None, timeout=None, grace=30):
            commands.append(command)
            path = self.root / f'pids-{len(commands)}'
            pids.append(path)
            # Real process tree whose grandchild ignores SIGTERM; grace is the real AGENT_GRACE.
            return swe_io.execute([sys.executable, '-c', TREE, str(path)], self.root, log, timeout=timeout, grace=grace)

        def fire():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if len(pids) == 2 and all(p.exists() and p.read_text() for p in pids):
                    break
                time.sleep(0.05)
            signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
        threading.Thread(target=fire, daemon=True).start()
        rows = [row(t['id']) for t in tasks(3)]
        started = time.monotonic()
        with self.assertRaises(KeyboardInterrupt), patch('swe_eval.runner.worker'), \
                patch('swe_eval.runner.environment_python', return_value=sys.executable), \
                patch('swe_eval.runner.execute', side_effect=fake_execute), patch('sys.stdout', io.StringIO()):
            run_swe(make_config(suite={'concurrency': 2}), tasks(3), self.output, False, rows)
        self.assertLess(time.monotonic() - started, 8, 'interrupt must not wait for the 60s grace')
        processes = [int(x) for p in pids for x in p.read_text().split()]
        deadline = time.monotonic() + 5
        while any(alive(pid) for pid in processes) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse([pid for pid in processes if alive(pid)])
        # Only the two agents ran: the queued task never started and nothing was scored.
        self.assertEqual(len(commands), 2)
        self.assertTrue(all('swe-agent' in c for c in commands))
        self.assertEqual([r['status'] for r in rows], ['pending'] * 3)
        self.assertFalse(swe_io._children)
        self.assertFalse(swe_io._halted.is_set(), 'later runs in the same process must start normally')

    def test_concurrency_validation_and_harbor_trials(self):
        for value in (0, 5, '2', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_config(suite={'concurrency': value})
        config = make_config(suite={'concurrency': 2})
        self.assertEqual(tb_config(config, [], self.root, False)['n_concurrent_trials'], 2)
        self.assertEqual(make_config().concurrency, 1)
        self.assertNotIn('concurrency', config.comparison_settings()['suite'])


if __name__ == '__main__':
    unittest.main()
