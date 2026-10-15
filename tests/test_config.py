import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from swe_eval.__main__ import main
from swe_eval.config import Config, load_env
from swe_eval.doctor import check_api
from swe_eval.runner import tb_config

SUITE = '''benchmarks = ["swebench"]
max_tokens = 4096
max_turns = 10
swe_agent_timeout = 600
swe_score_timeout = 600
api_timeout = 60
[manifests.swebench]
name = "smoke"
ids = ["a__a-1"]
'''
BASE = '''model = "base-model"
checkpoint = "base"
base_url = "http://127.0.0.1:8000/v1"
server = "vllm"
model_path = "/models/base"
vllm_version = "0.31.0"
context_length = 65536
reasoning = true
[sampling]
temperature = 0.6
top_p = 0.95
[extra_body]
top_k = 20
chat_template_kwargs = { enable_thinking = true }
'''


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'models').mkdir()
        (self.root / 'suites').mkdir()
        self.write('suites/suite.toml', SUITE)
        self.write('models/base.toml', BASE)
        configs = patch('swe_eval.config.CONFIGS', self.root)
        configs.start()
        self.addCleanup(configs.stop)

    def write(self, name, text):
        (self.root / name).write_text(text)

    def test_extends_and_explicit_override(self):
        self.write('models/ckpt.toml', 'extends = "base"\nmodel = "ckpt-2000"\ncheckpoint = "/ckpt/2000"\n'
                   '[sampling]\ntemperature = 0.0\n')
        config = Config.load('ckpt', 'suite', ['max_turns=3', 'extra_body.top_k=5', 'checkpoint=step 2000'])
        self.assertEqual((config.model, config.model_path, config.server), ('ckpt-2000', '/models/base', 'vllm'))
        # Tables merge per key: child changes temperature, top_p is inherited.
        self.assertEqual(config.sampling_params, {'temperature': 0.0, 'top_p': 0.95})
        self.assertEqual(config.max_turns, 3)
        self.assertEqual(config.extra_body['top_k'], 5)
        self.assertEqual(config.extra_body['chat_template_kwargs'], {'enable_thinking': True})
        self.assertEqual(config.checkpoint, 'step 2000')
        snapshot = config.snapshot()
        self.assertEqual(snapshot['overrides'], {'max_turns': 3, 'extra_body.top_k': 5, 'checkpoint': 'step 2000'})
        self.assertEqual(Config.from_snapshot(json.loads(json.dumps(snapshot))), config)

    def test_extends_is_single_level(self):
        self.write('models/mid.toml', 'extends = "base"\n')
        self.write('models/leaf.toml', 'extends = "mid"\n')
        with self.assertRaisesRegex(ValueError, 'single level'):
            Config.load('leaf', 'suite')

    def test_unknown_fields_rejected(self):
        cases = [('models/bad.toml', 'api_key = "inline"\n' + BASE, 'Unknown model fields'),
                 ('models/bad.toml', BASE.replace('top_p = 0.95', 'top_k = 20'), 'Unknown sampling fields'),
                 ('suites/bad.toml', 'max_retries = 3\n' + SUITE, 'Unknown suite fields')]
        for name, text, message in cases:
            self.write(name, text)
            with self.subTest(name=name, message=message), self.assertRaisesRegex(ValueError, message):
                Config.load('bad', 'bad' if name.startswith('suites') else 'suite')
        with self.assertRaisesRegex(ValueError, 'Unknown --set field'):
            Config.load('base', 'suite', ['temperature=0'])

    def test_invalid_values_rejected(self):
        for override in ['max_tokens=65536', 'base_url=https://user:pw@host/v1', 'sampling.top_p=nan',
                         'max_turns=0', 'server=sglang', 'swe_scoring=always', 'extra_body.max_tokens=1',
                         'max_turns="3"']:
            with self.subTest(override=override), self.assertRaises(ValueError):
                Config.load('base', 'suite', [override])

    def test_api_key_reads_only_declared_variable(self):
        self.write('models/hosted.toml', BASE.replace('server = "vllm"', 'server = "other"\napi_key_env = "HOSTED_KEY"'))
        config = Config.load('hosted', 'suite')
        with patch.dict(os.environ, {'EVAL_API_KEY': 'legacy', 'OPENAI_API_KEY': 'ambient'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'HOSTED_KEY is not set'):
                config.api_key()
            with patch('swe_eval.doctor.request') as request, self.assertRaises(ValueError):
                check_api(config)
            request.assert_not_called()
            # Helper processes never inherit an ambient key.
            self.assertNotIn('OPENAI_API_KEY', config.child_env(api=False))
        with patch.dict(os.environ, {'HOSTED_KEY': 'hosted-secret'}, clear=True):
            self.assertEqual(config.child_env()['OPENAI_API_KEY'], 'hosted-secret')
            self.assertNotIn('hosted-secret', json.dumps(config.snapshot()))
        # Unauthenticated endpoint is declared by an empty api_key_env, never by fallback.
        self.assertEqual(Config.load('base', 'suite').api_key(), 'EMPTY')

    def test_env_file_reads_credentials_only(self):
        env = self.root / '.env'
        env.write_text('EVAL_MODEL=ignored\nEVAL_API_KEY=legacy\nVLLM_VERSION=1\nMODEL_API_KEY="$(touch bad)"\n'
                       'EVAL_SWE_REVISION=abc\n')
        with patch.dict(os.environ, {}, clear=True):
            ignored = load_env(env)
            self.assertEqual(ignored, ['EVAL_MODEL', 'EVAL_API_KEY', 'VLLM_VERSION'])
            self.assertEqual(os.environ['MODEL_API_KEY'], '$(touch bad)')
            self.assertEqual(os.environ['EVAL_SWE_REVISION'], 'abc')
            self.assertNotIn('EVAL_MODEL', os.environ)

    def test_non_vllm_reasoning_model_extra_body(self):
        self.write('models/hosted.toml', '''model = "provider/reasoner"
base_url = "https://api.example.com/v1"
api_key_env = "HOSTED_KEY"
server = "other"
context_length = 131072
reasoning = true
[sampling]
temperature = 1.0
[extra_body]
reasoning_effort = "medium"
''')
        config = Config.load('hosted', 'suite')
        self.assertEqual(config.sampling(), {'temperature': 1.0, 'max_tokens': 4096,
                                             'extra_body': {'reasoning_effort': 'medium'}})
        calls = []

        def respond(config, endpoint, payload=None, server_root=False):
            calls.append((endpoint, payload))
            content = '```bash\necho 323\n```' if len(calls) == 1 else json.dumps({
                'analysis': '', 'plan': '', 'commands': [{'keystrokes': 'echo 324\n', 'duration': 1}],
                'task_complete': False})
            return {'choices': [{'finish_reason': 'stop',
                                 'message': {'content': content, 'reasoning_content': 'thinking'}}]}
        with patch.dict(os.environ, {'HOSTED_KEY': 'hosted-test-key'}), patch('swe_eval.doctor.request', side_effect=respond):
            result = check_api(config)
        self.assertEqual([c[0] for c in calls], ['/chat/completions'] * 2)
        self.assertEqual(calls[0][1]['reasoning_effort'], 'medium')
        self.assertNotIn('chat_template_kwargs', calls[0][1])
        self.assertNotIn('top_k', calls[0][1])
        self.assertTrue(result['reasoning_validated'])
        self.assertFalse(result['server_context_verified'])
        agent = tb_config(config, [], self.root, False)['agents'][0]['kwargs']
        self.assertEqual(agent['llm_call_kwargs']['extra_body'], {'reasoning_effort': 'medium'})

    def test_snapshot_detects_changed_template(self):
        template = self.root / 'template.jinja'
        template.write_text('v1')
        self.write('models/templated.toml', f'chat_template = "{template.as_posix()}"\n' + BASE)
        snapshot = Config.load('templated', 'suite').snapshot()
        template.write_text('v2')
        with self.assertRaisesRegex(ValueError, 'chat_template changed'):
            Config.from_snapshot(snapshot)


class RepositoryConfigTests(unittest.TestCase):
    def test_shipped_configs_load(self):
        for model in ('qwen35-base', 'qwen35-sft.example', 'openai-compatible.example'):
            for suite in ('smoke', 'small-v1', 'full'):
                with self.subTest(model=model, suite=suite):
                    config = Config.load(model, suite)
                    self.assertLess(config.max_tokens, config.context_length)
        base = Config.load('qwen35-base', 'smoke')
        self.assertEqual(base.extra_body['chat_template_kwargs'], {'enable_thinking': True})
        self.assertTrue(base.reasoning)
        self.assertEqual(len(base.chat_template_sha256), 64)
        self.assertEqual(Config.load('qwen35-sft.example', 'smoke').model_path, '/root/ckpt/sft/step-2000')

    def test_cli_requires_model_for_api_doctor(self):
        with patch('swe_eval.__main__.load_env', return_value=[]), \
                patch('swe_eval.__main__.doctor') as doctor, patch('sys.stderr'):
            self.assertEqual(main(['doctor']), 2)
            doctor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
