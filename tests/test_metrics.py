import json
from pathlib import Path
import tempfile
import unittest

from swe_eval.io import read_json, write_json
from swe_eval.metrics import enrich
from swe_eval.results import row, summarize
from swe_eval.telemetry import Recorder


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.task = self.output / 'tasks' / 'a'
        self.recorder = Recorder(self.task)

    def response(self, index, *, error='', recovered=False, tools=0):
        r = self.recorder
        r.response_index = index
        r.emit('llm_start', messages=[])
        r.emit('llm_response', response={'usage': {'prompt_tokens': 10, 'completion_tokens': 5}})
        r.emit('action_parse', raw_error=error, error='' if recovered else error)
        for _ in range(tools):
            r.emit('tool_start', command='echo example')
            r.emit('tool_end', result=None)

    def finish(self, result=None):
        self.recorder.emit('agent_end')
        self.recorder.emit('container_state', oom_killed=False)
        return enrich(result or row('a', 'task_failed'), self.task, self.output)

    def test_batch_commands_rounds_and_raw_format_error_micro_rate(self):
        self.response(1, error='invalid JSON')
        self.response(2, error='autocorrected JSON', recovered=True, tools=2)
        self.response(3, tools=1)
        result = self.finish()
        self.assertEqual(result['agent_rounds'], 3)
        self.assertEqual(result['tool_rounds'], 2)
        self.assertEqual(result['tool_call_count'], 3)
        self.assertEqual(result['tool_format_error_count'], 2)
        self.assertEqual(result['unrecoverable_format_error_count'], 1)
        self.assertEqual(result['format_auto_correction_count'], 1)
        self.assertAlmostEqual(result['tool_format_error_rate'], 2/3)
        self.assertEqual(result['output_tokens'], 15)
        self.assertTrue(result['trace_complete'])
        self.assertTrue(result['metrics_complete'])
        summary = summarize(self.output, [result], 1, 1)
        self.assertEqual(summary['pass_at_1'], 0)
        self.assertEqual(summary['resolved_rate'], 0)
        self.assertEqual(summary['avg_tool_rounds'], 2)
        self.assertAlmostEqual(summary['tool_format_error_rate'], 2/3)
        self.assertEqual(summary['oom_observed_rate'], 0)
        self.assertTrue((self.output / 'failures.json').is_file())

    def test_agent_budget_timeout_separate_from_infrastructure(self):
        self.response(1, tools=1)
        result = self.finish(row('a', 'budget_exhausted', 'AgentTimeoutError: 900 seconds'))
        summary = summarize(self.output, [result], 1, 1)
        self.assertEqual(summary['timeout_rate'], 1)
        self.assertEqual(summary['agent_timeout_rate'], 1)
        self.assertEqual(summary['infrastructure_timeout_rate'], 0)
        self.assertTrue(summary['complete'])

    def test_tool_and_api_timeouts_observed_separately(self):
        self.response(1, tools=1)
        self.recorder.emit('tool_error', exception_type='TimeoutExpired', detail='command timed out')
        self.recorder.emit('llm_error', exception_type='APITimeoutError', detail='request timed out')
        result = self.finish(row('a', 'api_error'))
        self.assertEqual(result['timeout_stages'], ['api', 'tool'])
        summary = summarize(self.output, [result], 1, 1)
        self.assertEqual(summary['infrastructure_timeout_rate'], 1)
        self.assertIsNone(summary['pass_at_1'])

    def test_partial_event_stream_and_missing_usage_not_assumed_zero(self):
        self.recorder.emit('llm_start')
        with self.recorder.path.open('a') as file:
            file.write('{"incomplete"')
        result = enrich(row('a', 'budget_exhausted'), self.task, self.output)
        self.assertFalse(result['trace_complete'])
        self.assertFalse(result['token_usage_complete'])
        self.assertIsNone(result['output_tokens'])
        self.assertFalse(result['metrics_complete'])
        self.assertIsNone(result['tool_format_error_rate'])

    def test_confirmed_oom_is_infrastructure_exit_137_is_not_proof(self):
        self.response(1, tools=1)
        self.recorder.emit('tool_end', result={'returncode': 137})
        result = self.finish()
        self.assertFalse(result['oom_observed'])
        self.recorder.emit('container_state', oom_killed=True)
        result = enrich(row('a', 'task_failed'), self.task, self.output)
        self.assertTrue(result['oom_observed'])
        self.assertEqual(result['status'], 'environment_error')
        summary = summarize(self.output, [result], 1, 1)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['oom_observed_rate'], 1)

    def test_missing_metrics_retains_unknown_coverage(self):
        result = enrich(row('b', 'environment_error'), self.output / 'absent', self.output)
        summary = summarize(self.output, [result], 1, 1)
        self.assertIsNone(summary['avg_tool_rounds'])
        self.assertIsNone(summary['timeout_rate'])
        self.assertIsNone(summary['oom_observed_rate'])
        self.assertFalse(summary['metrics_complete'])

    def test_summary_micro_rate_and_average_denominators(self):
        a, b = row('a', 'task_failed'), row('b', 'task_failed')
        a.update(tool_rounds=2, action_parse_attempts=2, tool_format_error_count=1)
        b.update(tool_rounds=8, action_parse_attempts=8, tool_format_error_count=1)
        summary = summarize(self.output, [a,b], 1, 2)
        self.assertEqual(summary['avg_tool_rounds'], 5)
        self.assertEqual(summary['tool_format_error_rate'], .2)


if __name__ == '__main__':
    unittest.main()
