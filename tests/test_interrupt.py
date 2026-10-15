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

from swe_eval.config import digest
from swe_eval.io import execute, read_json, register_secret, write_json
from swe_eval.runner import cleanup_containers, run, run_suite
from tests.test_framework import make_config

# Parent spawns a child that ignores SIGTERM; both write their PIDs, then sleep.
TREE = '''
import os, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c',
    'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
open(sys.argv[1], 'w').write(f'{os.getpid()} {child.pid}')
print('line with interrupt-test-hidden-value', flush=True)
time.sleep(60)
'''


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # Reaped zombies are gone; unreaped zombies of other parents count as dead.
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except OSError:
        return False


class InterruptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def interrupt_after(self, path, delay=0.2):
        def fire():
            while not path.exists() or not path.read_text():
                time.sleep(0.05)
            time.sleep(delay)
            signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
        threading.Thread(target=fire, daemon=True).start()

    def test_interrupt_kills_whole_group_and_keeps_live_redacted_log(self):
        pids, log = self.root / 'pids', self.root / 'live.log'
        self.interrupt_after(pids)
        register_secret('interrupt-test-hidden-value')  # As config.api_key() does for real runs.
        with self.assertRaises(KeyboardInterrupt):
            execute([sys.executable, '-c', TREE, str(pids)], self.root, log, grace=0.5)
        parent, child = map(int, pids.read_text().split())
        deadline = time.monotonic() + 5
        while (alive(parent) or alive(child)) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(alive(parent))
        self.assertFalse(alive(child), 'SIGTERM-ignoring grandchild must be SIGKILLed after grace')
        self.assertEqual(log.read_text(), 'line with [REDACTED]\n')

    def test_log_is_written_while_running(self):
        log = self.root / 'live.log'
        seen = []
        def watch():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if log.exists() and 'first' in log.read_text():
                    seen.append(True)
                    return
                time.sleep(0.05)
        watcher = threading.Thread(target=watch)
        watcher.start()
        execute([sys.executable, '-c', 'import time; print("first", flush=True); time.sleep(1); print("second")'],
                self.root, log)
        watcher.join()
        self.assertEqual(seen, [True])
        self.assertEqual(log.read_text(), 'first\nsecond\n')

    def manifest(self):
        tasks = [{'id': f't{i}', 'record': {'instance_id': f't{i}', 'patch': 'gold', 'problem_statement': 'x'}}
                 for i in range(3)]
        return {'benchmark': 'swebench', 'profile': 'small', 'tasks': tasks, 'tasks_sha256': digest(tasks)}

    def test_interrupted_run_marks_stopped_cleans_up_and_summarizes(self):
        lock = self.root / 'environments' / 'swebench' / 'uv.lock'
        lock.parent.mkdir(parents=True)
        lock.write_text('locked')
        started = self.root / 'second-task-started'

        def fake_worker(benchmark, args, log, **kwargs):
            if args[0] == 'metadata':
                write_json(args[2], {'versions': {}, 'all_distributions': [], 'python': 'test',
                                     'agent_sources_sha256': 'source'})

        def fake_execute(command, cwd, log, env=None, timeout=None, grace=30):
            target = Path(command[-1])
            task_id = target.parent.name
            if task_id == 't1':
                started.write_text('1')
                time.sleep(10)  # Interrupted here.
            write_json(target / task_id / f'{task_id}.traj.json', {'info': {'exit_status': 'LimitsExceeded'}})
            return {'elapsed_seconds': 1, 'returncode': 0, 'timed_out': False}

        self.interrupt_after(started, delay=0.1)
        config = make_config(suite={'manifests': {'swebench': 'small', 'tbv2': 'small'}})
        with patch.dict(os.environ, {'EVAL_TEST_KEY': 'interrupt-run-test-key'}), patch('swe_eval.runner.ROOT', self.root), patch('swe_eval.runner.prepare', return_value=self.manifest()), \
                patch('swe_eval.runner.worker', side_effect=fake_worker), \
                patch('swe_eval.runner.doctor', return_value={'ok': True, 'checks': {}}), \
                patch('swe_eval.runner.environment_python', return_value=sys.executable), \
                patch('swe_eval.runner.execute', side_effect=fake_execute), \
                patch('swe_eval.runner.docker_oom_events', return_value=None), \
                patch('swe_eval.runner.cleanup_containers') as cleanup, patch('sys.stdout', io.StringIO()):
            results = run_suite(config, self.root / 'runs')
        # The interrupt stops the suite: tbv2 never starts.
        self.assertEqual([r['benchmark'] for r in results], ['swebench'])
        summary = results[0]
        output = Path(summary['output'])
        cleanup.assert_called_once_with(output, output.name)
        self.assertTrue(summary['stopped'])
        self.assertFalse(summary['complete'])
        self.assertIsNone(summary['success_rate'])
        self.assertEqual(summary['status_counts'], {'budget_exhausted': 1, 'stopped': 2})
        rows = read_json(output / 'results.json')
        self.assertEqual([r['status'] for r in rows], ['budget_exhausted', 'stopped', 'stopped'])
        self.assertTrue(read_json(output / 'summary.json')['stopped'])

    def test_cleanup_filters_only_this_run(self):
        harbor = self.root / 'harbor' / 'job' / 'Fix-Git__AbC1234'
        harbor.mkdir(parents=True)
        listings = {
            ('ps', '-aq', '--filter', 'label=swe_eval.run=my-run'): ['c-agent', ''],
            ('ps', '-a', '--filter', 'name=.my-run', '--format', '{{.ID}} {{.Names}}'):
                ['c-score sweb.eval.a__a-1.my-run', 'c-other sweb.eval.a__a-1.my-run-2', ''],
            ('ps', '-a', '--format', '{{.ID}} {{.Label "com.docker.compose.project"}}'):
                ['c-tb fix-git__abc1234__env', 'c-foreign other-trial__env', 'c-plain ', ''],
            ('network', 'ls', '--format', '{{.ID}} {{.Label "com.docker.compose.project"}}'):
                ['n-tb fix-git__abc1234__env', 'n-bridge ', ''],
        }
        calls = []
        def fake_docker(*args):
            calls.append(args)
            return listings.get(args, [])
        with patch('swe_eval.runner.docker_lines', side_effect=fake_docker):
            removed = cleanup_containers(self.root, 'my-run')
        self.assertEqual(removed['containers'], ['c-agent', 'c-score', 'c-tb'])
        self.assertEqual(removed['networks'], ['n-tb'])
        self.assertIn(('rm', '-f', *removed['containers']), calls)
        self.assertIn(('network', 'rm', 'n-tb'), calls)
        self.assertEqual(read_json(self.root / 'cleanup.json')['containers'], removed['containers'])

    def test_suite_runs_every_benchmark_with_default_ids(self):
        names = []
        def fake_run(benchmark, run_id, config, runs_dir, oracle):
            names.append(run_id)
            return runs_dir / run_id, {'complete': True, 'stopped': False}
        with patch('swe_eval.runner.run', side_effect=fake_run):
            results = run_suite(make_config(), self.root)
        self.assertEqual([r['benchmark'] for r in results], ['swebench', 'tbv2'])
        self.assertTrue(all(n.startswith('test-model__test-suite__') for n in names))
        self.assertEqual(names[0].split('__')[2], 'swebench')
        with self.assertRaises(ValueError):
            run_suite(make_config(), self.root, run_id='fixed')


if __name__ == '__main__':
    unittest.main()
