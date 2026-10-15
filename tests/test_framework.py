import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from swe_eval.config import Config, digest
from swe_eval.doctor import check_api
from swe_eval.io import execute, read_json, scrub_tree, write_json
from swe_eval.results import classify, mini_usage, row, summarize, swe_result, tb_results
from swe_eval.runner import run, tb_config, validate_manifest


SMOKE = {'name': 'smoke', 'ids': ['a__a-1']}


def make_config(model=None, suite=None):
    model = {'model': 'qwen35-9b', 'base_url': 'http://localhost:8000/v1', 'context_length': 131072,
             'server': 'vllm', 'checkpoint': 'baseline', 'model_path': '/model', 'vllm_version': 'test',
             'chat_template': 'models/templates/qwen35-9b.jinja', 'reasoning': True, 'api_key_env': 'EVAL_TEST_KEY',
             'sampling': {'temperature': 0.6, 'top_p': 0.95},
             'extra_body': {'top_k': 20, 'chat_template_kwargs': {'enable_thinking': True}}, **(model or {})}
    suite = {'benchmarks': ['swebench', 'tbv2'], 'manifests': {'swebench': SMOKE, 'tbv2': SMOKE},
             'max_tokens': 16384, 'max_turns': 100, 'swe_agent_timeout': 3600, 'swe_score_timeout': 1800,
             'api_timeout': 180, **(suite or {})}
    from swe_eval.config import check_model, check_suite
    return Config.build('test-model', check_model(model), 'test-suite', check_suite(suite), {})


HOSTED = {'model': 'example-coder', 'base_url': 'https://api.example.com/v1', 'server': 'other',
          'api_key_env': 'MODEL_API_KEY', 'reasoning': False, 'model_path': '', 'vllm_version': '',
          'extra_body': {}}


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = make_config()
        environment = patch.dict(os.environ, {'EVAL_TEST_KEY': 'test-key'})
        environment.start()
        self.addCleanup(environment.stop)

    def manifest(self, benchmark='swebench'):
        tasks = [{'id': 'a__a-1', 'record': {'instance_id': 'a__a-1', 'patch': 'gold', 'problem_statement': 'fix'}}]
        return {'schema_version': 2, 'benchmark': benchmark, 'name': 'smoke', 'spec': SMOKE,
                'tasks': tasks, 'tasks_sha256': digest(tasks)}

    def test_hosted_preflight_skips_vllm_metadata_but_checks_actions(self):
        with patch.dict(os.environ, {'MODEL_API_KEY': 'private-key'}, clear=True):
            config = make_config(HOSTED)
            calls = []
            def respond(config, endpoint, payload):
                calls.append((endpoint, payload))
                content = '```bash\necho 323\n```' if len(calls) == 1 else json.dumps({
                    'analysis': '', 'plan': '', 'commands': [{'keystrokes': 'echo 324\n', 'duration': 1}],
                    'task_complete': False})
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}
            with patch('swe_eval.doctor.request', side_effect=respond):
                result = check_api(config)
        self.assertEqual([c[0] for c in calls], ['/chat/completions', '/chat/completions'])
        self.assertFalse(result['reasoning_validated'])
        self.assertFalse(result['server_context_verified'])
        self.assertNotIn('chat_template_kwargs', calls[0][1])
        self.assertNotIn('top_k', calls[0][1])

    def test_hosted_preflight_missing_key_fails_without_request(self):
        with patch.dict(os.environ, {'MODEL_API_KEY': ''}, clear=True):
            config = make_config(HOSTED)
            with patch('swe_eval.doctor.request') as request, self.assertRaises(ValueError):
                check_api(config)
            request.assert_not_called()

    def test_hosted_key_is_redacted(self):
        from swe_eval.io import redact
        with patch.dict(os.environ, {'MODEL_API_KEY': 'private-key'}, clear=True):
            self.assertEqual(redact('Bearer private-key'), 'Bearer [REDACTED]')

    def test_reasoning_multiturn_wire_protocol(self):
        calls = []
        def respond(req, timeout):
            calls.append(req)
            if req.full_url.endswith('/models'):
                body = {'data': [{'id': self.config.model, 'root': '/model', 'max_model_len': 131072}]}
            elif req.full_url.endswith('/version'):
                body = {'version': 'test'}
            else:
                content = '```bash\necho 323\n```' if len(calls) == 3 else json.dumps({
                    'analysis': 'add one', 'plan': 'echo', 'commands': [{'keystrokes': 'echo 324\n', 'duration': 1}], 'task_complete': False})
                body = {'choices': [{'finish_reason': 'stop', 'message': {'content': content, 'reasoning_content': 'private reasoning'}}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}
            return io.BytesIO(json.dumps(body).encode())
        with patch.dict(os.environ, {'EVAL_TEST_KEY': 'secret'}), patch('swe_eval.doctor.urlopen', side_effect=respond):
            result = check_api(self.config)
        self.assertEqual(len(result['turns']), 2)
        first, second = [json.loads(c.data) for c in calls if c.data]
        self.assertTrue(first['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(first['max_tokens'], 16384)
        self.assertNotIn('tools', first)
        self.assertEqual(second['messages'][1], {'role': 'assistant', 'content': '```bash\necho 323\n```'})
        self.assertNotIn('private reasoning', json.dumps(second))
        self.assertEqual(calls[-1].get_header('Authorization'), 'Bearer secret')

    def test_doctor_rejects_unseparated_thinking(self):
        responses = [{'data': [{'id': self.config.model, 'max_model_len': 131072}]}, {'version': 'test'},
                     {'choices': [{'finish_reason': 'stop', 'message': {'content': '<think>oops</think>323'}}]}]
        with patch('swe_eval.doctor.request', side_effect=responses), self.assertRaises(ValueError):
            check_api(self.config)

    def test_doctor_context_and_version_mismatch(self):
        with patch('swe_eval.doctor.request', return_value={'data': [{'id': self.config.model, 'max_model_len': 4096}]}), self.assertRaises(ValueError):
            check_api(self.config)
        with patch('swe_eval.doctor.request', side_effect=[{'data': [{'id': self.config.model, 'max_model_len': 131072}]}, {'version': 'different'}]), self.assertRaises(ValueError):
            check_api(self.config)

    def test_manifest_integrity(self):
        manifest = self.manifest()
        validate_manifest(manifest, 'swebench', SMOKE)
        with self.assertRaisesRegex(ValueError, 'different spec'):
            validate_manifest(manifest, 'swebench', dict(SMOKE, ids=['b__b-1']))
        manifest['tasks'][0]['record']['patch'] = 'changed'
        with self.assertRaises(ValueError):
            validate_manifest(manifest, 'swebench', SMOKE)

    def test_subprocess_timeout_and_redaction(self):
        with patch.dict(os.environ, {'EVAL_API_KEY': 'do-not-record'}):
            log = self.root / 'process.log'
            result = execute([sys.executable, '-c', 'import time; print("do-not-record", flush=True); time.sleep(10)'],
                             self.root, log, timeout=0.1)
            self.assertTrue(result['timed_out'])
            self.assertNotIn('do-not-record', log.read_text())
            write_json(self.root / 'config.json', {'error': 'do-not-record'})
            self.assertNotIn('do-not-record', (self.root / 'config.json').read_text())

    def test_scrub_does_not_follow_symlinks(self):
        outside = self.root / 'outside'
        outside.write_text('secret')
        artifacts = self.root / 'artifacts'
        artifacts.mkdir()
        (artifacts / 'link').symlink_to(outside)
        with patch.dict(os.environ, {'EVAL_API_KEY': 'secret'}):
            scrub_tree(artifacts)
        self.assertEqual(outside.read_text(), 'secret')

    def test_error_classification(self):
        for error, status in [('AuthenticationError', 'api_error'), ('APIConnectionError', 'api_error'),
                              ('AgentTimeoutError', 'budget_exhausted'), ('LimitsExceeded', 'budget_exhausted'),
                              ('ContextWindowExceededError', 'budget_exhausted'), ('VerifierTimeoutError', 'environment_error'),
                              ('Docker permission denied', 'environment_error'), ('unknown', 'environment_error')]:
            self.assertEqual(classify(error), status)

    def test_mini_usage_no_double_count_reasoning(self):
        trajectory = {'messages': [{'extra': {'response': {'usage': {'prompt_tokens': 3, 'completion_tokens': 9,
                         'completion_tokens_details': {'reasoning_tokens': 7}}}}}]}
        self.assertEqual(mini_usage(trajectory), {'input_tokens': 3, 'output_tokens': 9})
        self.assertIsNone(mini_usage({})['input_tokens'])

    def test_harness_requires_actual_report(self):
        agent = self.root / 'agent'
        agent.mkdir()
        write_json(agent / 'a.traj.json', {'info': {'exit_status': 'Submitted'}, 'messages': []})
        process = {'elapsed_seconds': 1, 'timed_out': False, 'returncode': 0}
        score = self.root / 'score'
        self.assertEqual(swe_result('a', agent, score, process)['status'], 'environment_error')
        write_json(score / 'logs' / 'report.json', {'a': {'resolved': False}})
        self.assertEqual(swe_result('a', agent, score, process)['status'], 'task_failed')
        write_json(score / 'logs' / 'report.json', {'a': {'resolved': True}})
        self.assertTrue(swe_result('a', agent, score, process)['resolved'])

    def test_official_empty_patch_and_apply_failure_are_model_failures(self):
        agent = self.root / 'agent'
        agent.mkdir()
        write_json(agent / 'a.traj.json', {'info': {'exit_status': 'Submitted'}, 'messages': []})
        process = {'elapsed_seconds': 1, 'timed_out': False, 'returncode': 0}
        score = self.root / 'score'
        write_json(score / 'candidate.score.json', {'empty_patch_ids': ['a'], 'error_ids': []})
        result = swe_result('a', agent, score, process)
        self.assertEqual(result['status'], 'task_failed')
        self.assertIs(result['resolved'], False)
        (score / 'candidate.score.json').unlink()
        log = score / 'logs' / 'a' / 'run_instance.log'
        log.parent.mkdir(parents=True)
        log.write_text('2026-10-13 23:25:21 - INFO - >>>>> Patch Apply Failed:\ncorrupt patch\n')
        self.assertEqual(swe_result('a', agent, score, process)['status'], 'task_failed')

    def test_native_turn_budget_keeps_official_empty_submission_verdict(self):
        agent = self.root / 'agent'
        agent.mkdir()
        write_json(agent / 'a.traj.json', {'info': {'exit_status': 'LimitsExceeded'}, 'messages': []})
        score = self.root / 'score'
        process = {'elapsed_seconds': 1, 'timed_out': False, 'returncode': 0}
        result = swe_result('a', agent, score, process)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertIsNone(result['resolved'])
        write_json(score / 'candidate.score.json', {'empty_patch_ids': ['a']})
        result = swe_result('a', agent, score, process)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertIs(result['resolved'], False)

    def test_harbor_trial_reward_and_missing_task(self):
        tasks = [{'id': 'a'}, {'id': 'b'}]
        write_json(self.root / 'job' / 'trial' / 'result.json', {
            'task_name': 'a', 'finished_at': '2026-10-13T01:00:01+00:00',
            'started_at': '2026-10-13T01:00:00+00:00',
            'agent_result': {'n_input_tokens': 12, 'n_output_tokens': 20},
            'verifier_result': {'rewards': {'reward': 1}}})
        results = tb_results(tasks, self.root)
        self.assertEqual([r['status'] for r in results], ['success', 'environment_error'])
        summary = summarize(self.root, results, 3, 2)
        self.assertFalse(summary['complete'])
        self.assertIsNone(summary['success_rate'])
        self.assertEqual(summary['task_count'], 2)

    def test_harbor_budget_classification(self):
        write_json(self.root / 'trial' / 'result.json', {
            'task_name': 'a', 'finished_at': '2026-10-13T01:00:01+00:00',
            'config': {'agent': {'kwargs': {'max_turns': 100}}},
            'agent_result': {'metadata': {'n_episodes': 100}},
            'verifier_result': {'rewards': {'reward': 0}}})
        self.assertEqual(tb_results([{'id': 'a'}], self.root)[0]['status'], 'budget_exhausted')

    def test_harbor_timeout_preserves_completed_verifier_reward(self):
        write_json(self.root / 'trial' / 'result.json', {
            'task_name': 'a', 'finished_at': '2026-10-13T01:00:01+00:00',
            'exception_info': {'exception_type': 'AgentTimeoutError', 'exception_message': '900 seconds'},
            'verifier_result': {'rewards': {'reward': 0.0}}})
        result = tb_results([{'id': 'a'}], self.root)[0]
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertIs(result['resolved'], False)

    def test_tb_config_native_timeouts_and_text_mode(self):
        config = tb_config(self.config, [{'task_config': {'path': 'a'}}], self.root, False)
        self.assertEqual(config['n_attempts'], 1)
        self.assertEqual(config['n_concurrent_trials'], 1)
        agent = config['agents'][0]
        self.assertNotIn('override_timeout_sec', agent)
        self.assertEqual(agent['kwargs']['parser_name'], 'json')
        self.assertFalse(agent['kwargs']['interleaved_thinking'])
        self.assertTrue(agent['kwargs']['llm_call_kwargs']['extra_body']['chat_template_kwargs']['enable_thinking'])

    def test_run_without_dependencies_records_incomplete(self):
        with patch('swe_eval.runner.prepare', side_effect=RuntimeError('not installed')):
            folder, summary = run('swebench', 'missing', self.config, self.root)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['task_count'], 1)
        self.assertTrue((folder / 'error.json').exists())
        with self.assertRaises(FileExistsError):
            run('swebench', 'missing', self.config, self.root)

    def test_preflight_api_error_preserved_in_final_summary(self):
        lock = self.root / 'environments' / 'swebench' / 'uv.lock'
        lock.parent.mkdir(parents=True)
        lock.write_text('locked')
        def fake_worker(benchmark, args, log):
            write_json(args[2], {'versions': {}, 'all_distributions': [], 'python': 'test',
                                 'agent_sources_sha256': 'source'})
        with patch('swe_eval.runner.ROOT', self.root), patch('swe_eval.runner.prepare', return_value=self.manifest()), \
                patch('swe_eval.runner.worker', side_effect=fake_worker), \
                patch('swe_eval.runner.doctor', return_value={'ok': False, 'checks': {'docker': {'ok': True}, 'api': {'ok': False}}}):
            output, summary = run('swebench', 'api-error', self.config, self.root / 'runs')
        self.assertFalse(summary['complete'])
        self.assertEqual(read_json(output / 'results.json')[0]['status'], 'api_error')
        self.assertEqual(summary['status_counts'], {'api_error': 1})

    def test_result_csv_redacts_error_credentials(self):
        result = row('a', 'api_error', 'API rejected private-api-key')
        with patch.dict(os.environ, {'EVAL_API_KEY': 'private-api-key'}):
            summarize(self.root, [result], 1, 1)
        self.assertNotIn('private-api-key', (self.root / 'results.csv').read_text())

    def test_duplicate_harbor_trial_is_incomplete(self):
        trial = {'task_name': 'a', 'finished_at': '2026-10-13T01:00:01+00:00',
                 'verifier_result': {'rewards': {'reward': 1}}}
        write_json(self.root / 'first' / 'result.json', trial)
        write_json(self.root / 'second' / 'result.json', trial)
        records = tb_results([{'id': 'a'}], self.root)
        self.assertEqual(records[0]['status'], 'environment_error')
        self.assertFalse(summarize(self.root, records, 1, 1)['complete'])

    def fake_swe(self, config, exit_status='Submitted', candidate='diff', resolved=False):
        from swe_eval.runner import run_swe
        output = self.root / 'run'
        output.mkdir()
        calls = []
        def fake_execute(command, cwd, log, env=None, timeout=None, grace=30):
            calls.append(command)
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            Path(log).write_text('ok')
            if 'swe-agent' in command:
                target = Path(command[-1])
                write_json(target / 'a__a-1' / 'a__a-1.traj.json', {'info': {'exit_status': exit_status}})
                write_json(target / 'preds.json', {'a__a-1': {'instance_id': 'a__a-1',
                           'model_patch': 'diff' if exit_status == 'Submitted' else ''}})
                (target / 'candidate.patch').write_text(candidate)
            else:
                predictions = read_json(command[command.index('--predictions_path') + 1])
                if predictions['a__a-1']['model_patch']:
                    write_json(Path(cwd) / 'logs' / 'report.json', {'a__a-1': {'resolved': resolved}})
                else:  # Official harness reports empty predictions only in its aggregate.
                    write_json(Path(cwd) / 'candidate.score.json', {'empty_patch_ids': ['a__a-1']})
            return {'elapsed_seconds': 1, 'returncode': 0, 'timed_out': False}
        tasks = self.manifest()['tasks']
        rows = [row(t['id']) for t in tasks]
        with patch('swe_eval.runner.worker'), patch('swe_eval.runner.environment_python', return_value=sys.executable), \
                patch('swe_eval.runner.execute', side_effect=fake_execute), patch('sys.stdout', io.StringIO()):
            run_swe(config, tasks, output, False, rows)
        return output, rows, calls

    def test_swe_orchestration_agent_to_harness(self):
        output, records, calls = self.fake_swe(self.config)
        self.assertEqual(records[0]['status'], 'task_failed')
        self.assertIsNone(records[0]['scored_from'])
        self.assertEqual(len(calls), 2)
        self.assertIn('swebench.harness.run_evaluation', calls[1])
        # Harness run_id is the run directory name, which names its containers for cleanup.
        self.assertEqual(calls[1][calls[1].index('--run_id') + 1], 'run')
        dataset_arg = calls[1][calls[1].index('--dataset_name') + 1]
        self.assertEqual(read_json(dataset_arg)[0]['patch'], 'gold')
        self.assertEqual(read_json(output / 'tasks' / 'a__a-1' / 'predictions.json')['a__a-1']['model_name_or_path'], 'candidate')

    def test_submitted_only_ignores_candidate_patch(self):
        _, records, calls = self.fake_swe(self.config, exit_status='LimitsExceeded', resolved=True)
        self.assertEqual(records[0]['status'], 'budget_exhausted')
        self.assertIs(records[0]['resolved'], False)
        self.assertIsNone(records[0]['scored_from'])
        # Native empty submission is still graded; the candidate patch is not.
        predictions = read_json(self.root / 'run' / 'tasks' / 'a__a-1' / 'predictions.json')
        self.assertEqual(predictions['a__a-1']['model_patch'], '')
        self.assertEqual(len(calls), 2)

    def test_final_workspace_grades_candidate_patch(self):
        config = make_config(suite={'swe_scoring': 'final-workspace'})
        output, records, calls = self.fake_swe(config, exit_status='LimitsExceeded', candidate='diff --git a', resolved=True)
        self.assertEqual(records[0]['status'], 'budget_exhausted')
        self.assertIs(records[0]['resolved'], True)
        self.assertEqual(records[0]['scored_from'], 'candidate')
        self.assertEqual(read_json(output / 'tasks' / 'a__a-1' / 'predictions.json')['a__a-1']['model_patch'], 'diff --git a')
        # An empty final workspace is not graded as a candidate.
        self.tmp.cleanup()
        self.root.mkdir()
        _, records, _ = self.fake_swe(config, exit_status='LimitsExceeded', candidate='', resolved=True)
        self.assertIsNone(records[0]['scored_from'])


if __name__ == '__main__':
    unittest.main()
