from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from swe_eval import runner
from swe_eval.io import write_json


class PullTests(unittest.TestCase):
    def test_pulls_only_missing_images_and_reports_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'profiles').mkdir()
            images = {'present': 'img/present:1', 'missing': 'img/missing:1', 'broken': 'img/broken:1', 'built': None}

            def fake_worker(benchmark, args, log, timeout):
                self.assertEqual(args[:2], ['images', 'tbv2'])
                write_json(args[3], images)

            def fake_run(command, **kwargs):
                self.assertEqual(command[:3], ['docker', 'pull', '-q'])
                if command[3] == 'img/broken:1':
                    return subprocess.CompletedProcess(command, 1, '', 'error: manifest unknown\n')
                return subprocess.CompletedProcess(command, 0, 'ok\n', '')

            with patch.object(runner, 'ROOT', root), patch.object(runner, 'prepare'), \
                    patch.object(runner, 'worker', side_effect=fake_worker), \
                    patch.object(runner, 'docker_lines', side_effect=lambda *a: ['sha'] if a[-1] == 'img/present:1' else None), \
                    patch.object(runner.subprocess, 'run', side_effect=fake_run) as pulls:
                rows = runner.pull_images('tbv2', {'name': 'small'})
        status = {r['id']: r['status'] for r in rows}
        self.assertEqual(status, {'present': 'present', 'missing': 'pulled',
                                  'broken': 'failed: error: manifest unknown', 'built': 'builds_from_dockerfile'})
        self.assertEqual([c.args[0][3] for c in pulls.call_args_list], ['img/missing:1', 'img/broken:1'])


if __name__ == '__main__':
    unittest.main()
