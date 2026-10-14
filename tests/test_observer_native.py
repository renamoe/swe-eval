"""Run in each isolated benchmark environment; no external API or Docker required."""
import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from swe_eval.metrics import enrich
from swe_eval.results import row
from swe_eval.telemetry import observe_mini, observe_harbor


@unittest.skipUnless(importlib.util.find_spec('minisweagent'), 'Requires SWE environment')
class MiniNativeObserverTest(unittest.TestCase):
    def test_native_format_recovery_and_submission_with_checkpoint(self):
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.run.utils.save import save_traj
        class Model:
            config = {}
            cost = 0
            n_calls = 0
            def get_template_vars(self):
                return {}
            def query(self, messages, **kwargs):
                self.n_calls += 1
                content = 'missing action' if self.n_calls == 1 else '```bash\necho submit\n```'
                return {'content': content, 'extra': {'response': {'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}}}
        class Environment:
            config = {}
            container_id = None
            def get_template_vars(self):
                return {}
            def execute(self, command, *args, **kwargs):
                return {'returncode': 0, 'output': 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nDONE'}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            directory = output / 'a'
            agent = DefaultAgent(Model(), Environment(), step_limit=3, cost_limit=0)
            def checkpoint():
                save_traj(agent, directory / 'a.traj.json', print_path=False, exit_status='Running')
            recorder = observe_mini(agent, directory, checkpoint)
            status, submission = agent.run('test')
            recorder.emit('agent_end')
            self.assertEqual(status, 'Submitted')
            self.assertEqual(submission, 'DONE')
            self.assertEqual(json.loads((directory/'a.traj.json').read_text())['messages'][-1]['content'], 'DONE')
            result = enrich(row('a','task_failed'), directory, output)
            self.assertEqual(result['agent_rounds'], 2)
            self.assertEqual(result['tool_rounds'], 1)
            self.assertEqual(result['tool_format_error_count'], 1)
            self.assertEqual(result['output_tokens'], 10)


@unittest.skipUnless(importlib.util.find_spec('harbor'), 'Requires Harbor environment')
class HarborNativeObserverTest(unittest.TestCase):
    def test_shipped_json_parser_and_async_capture(self):
        from harbor.agents.terminus_2.terminus_2 import Terminus2
        from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
        from harbor.llms.base import LLMResponse
        from harbor.models.metric import UsageInfo
        class LLM:
            async def call(self, *args, **kwargs):
                return LLMResponse(content=kwargs['prompt'], usage=UsageInfo(
                    prompt_tokens=10,completion_tokens=5,cache_tokens=0,cost_usd=0))
        class Session:
            async def send_keys(self, keys, *args, **kwargs):
                return None
        async def native_run(self, instruction, environment, context):
            bad = await self._llm.call(prompt='bad JSON')
            self._parser.parse_response(bad.content)
            good = await self._llm.call(prompt=json.dumps({'analysis':'echo','plan':'run',
                'commands':[{'keystrokes':'echo READY\n','duration':1}], 'task_complete':False}))
            parsed = self._parser.parse_response(good.content)
            await self._session.send_keys(parsed.commands[0].keystrokes)
            self._chat.messages.append({'role':'assistant','content':good.content})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            directory = output/'a__test'/'agent'
            agent = object.__new__(Terminus2)
            agent.logs_dir = directory
            agent._llm = LLM()
            agent._parser = TerminusJSONPlainParser()
            agent._session = Session()
            agent._chat = SimpleNamespace(messages=[])
            with patch.object(Terminus2,'run',native_run), patch('swe_eval.telemetry.subprocess.run',
                    return_value=subprocess.CompletedProcess([],0,stdout='',stderr='')):
                observe_harbor()
                asyncio.run(agent.run('test',SimpleNamespace(session_id='a__test__env'),None))
            result = enrich(row('a','task_failed'),directory,output)
            self.assertEqual(result['agent_rounds'],2)
            self.assertEqual(result['tool_call_count'],1)
            self.assertEqual(result['tool_format_error_count'],1)
            self.assertEqual(result['output_tokens'],10)
            self.assertTrue((directory/'messages.json').exists())


if __name__ == '__main__':
    unittest.main()
