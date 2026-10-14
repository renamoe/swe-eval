"""Observational hooks only: native prompts, parsing, retries and actions are unchanged."""
from dataclasses import asdict, is_dataclass
from functools import wraps
import json
from pathlib import Path
import subprocess
import time

from .io import redact, write_json

SCHEMA_VERSION = 2


def plain(value):
    if hasattr(value, 'model_dump'):
        return plain(value.model_dump(mode='json'))
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class Recorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'events.jsonl'
        self.response_index = 0
        self.emit('telemetry_start', schema_version=SCHEMA_VERSION)

    def emit(self, event, **data):
        record = {'event': event, 'timestamp': time.time(), 'response_index': self.response_index, **plain(data)}
        with self.path.open('a') as stream:
            stream.write(redact(json.dumps(record, ensure_ascii=False)) + '\n')


def inspect_containers(recorder, ids):
    if not ids:
        return
    try:
        command = subprocess.run(['docker', 'inspect', *ids], capture_output=True, text=True, timeout=10)
        if command.returncode:
            recorder.emit('container_probe_error', detail=command.stderr)
            return
        for container in json.loads(command.stdout):
            state = container.get('State', {})
            recorder.emit('container_state', container_id=container['Id'],
                          name=container.get('Name'), oom_killed=state.get('OOMKilled'),
                          exit_code=state.get('ExitCode'))
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        recorder.emit('container_probe_error', detail=str(exc))


def observe_mini(agent, directory, checkpoint):
    """Wrap one native agent instance with durable tracing; no subclass or policy changes."""
    from minisweagent.agents.default import FormatError
    recorder = Recorder(directory)
    recorder.emit('container_start', container_id=agent.env.container_id, task_id=Path(directory).name)
    original_add = agent.add_message
    original_query = agent.model.query
    original_parse = agent.parse_action
    original_execute = agent.env.execute

    @wraps(original_add)
    def add_message(*args, **kwargs):
        result = original_add(*args, **kwargs)
        checkpoint()
        return result

    @wraps(original_query)
    def query(messages, **kwargs):
        recorder.response_index += 1
        recorder.emit('llm_start', messages=messages, kwargs=kwargs)
        try:
            result = original_query(messages, **kwargs)
            recorder.emit('llm_response', response=result)
            return result
        except BaseException as exc:
            recorder.emit('llm_error', exception_type=type(exc).__name__, detail=str(exc))
            raise

    @wraps(original_parse)
    def parse(response):
        try:
            result = original_parse(response)
        except FormatError as exc:
            recorder.emit('action_parse', raw_error=str(exc), error=str(exc), command_count=0)
            raise
        recorder.emit('action_parse', raw_error='', error='', command_count=1)
        return result

    @wraps(original_execute)
    def execute(command, *args, **kwargs):
        recorder.emit('tool_start', command=command)
        try:
            result = original_execute(command, *args, **kwargs)
            recorder.emit('tool_end', result=result)
            return result
        except BaseException as exc:
            recorder.emit('tool_error', exception_type=type(exc).__name__, detail=str(exc))
            raise

    agent.add_message = add_message
    agent.model.query = query
    agent.parse_action = parse
    agent.env.execute = execute
    return recorder


def observe_harbor():
    """Install per-instance observers in the isolated Harbor worker process."""
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.environments.docker.docker import _sanitize_docker_compose_project_name
    original_run = Terminus2.run

    @wraps(original_run)
    async def run(self, instruction, environment, context):
        recorder = Recorder(self.logs_dir)
        recorder.emit('container_project', project=_sanitize_docker_compose_project_name(environment.session_id),
                      task_id=self.logs_dir.parent.name.rsplit('__', 1)[0])
        original_call = self._llm.call
        original_parse = self._parser.parse_response
        original_keys = self._session.send_keys

        @wraps(original_call)
        async def call(*args, **kwargs):
            recorder.response_index += 1
            recorder.emit('llm_start', args=args, kwargs=kwargs)
            try:
                result = await original_call(*args, **kwargs)
                recorder.emit('llm_response', response=result)
                return result
            except BaseException as exc:
                recorder.emit('llm_error', exception_type=type(exc).__name__, detail=str(exc),
                              truncated_response=getattr(exc, 'truncated_response', None))
                raise

        @wraps(original_parse)
        def parse(response):
            # The shipped JSON parser exposes a pure, pre-autocorrection validator.
            raw = self._parser._try_parse_response(response)
            result = original_parse(response)
            recorder.emit('action_parse', raw_error=raw.error, error=result.error,
                          warning=result.warning, command_count=len(result.commands))
            return result

        @wraps(original_keys)
        async def send_keys(keys, *args, **kwargs):
            recorder.emit('tool_start', command=keys)
            try:
                result = await original_keys(keys, *args, **kwargs)
                recorder.emit('tool_end', result=result)
                return result
            except BaseException as exc:
                recorder.emit('tool_error', exception_type=type(exc).__name__, detail=str(exc))
                raise

        self._llm.call = call
        self._parser.parse_response = parse
        self._session.send_keys = send_keys
        try:
            return await original_run(self, instruction, environment, context)
        except BaseException as exc:
            recorder.emit('agent_error', exception_type=type(exc).__name__, detail=str(exc))
            raise
        finally:
            recorder.emit('agent_end', messages=getattr(self._chat, 'messages', []))
            write_json(self.logs_dir / 'messages.json', getattr(self._chat, 'messages', []))
            project = _sanitize_docker_compose_project_name(environment.session_id)
            try:
                containers = subprocess.run(['docker', 'ps', '-aq', '--filter',
                                             'label=com.docker.compose.project=' + project],
                                            capture_output=True, text=True, timeout=10)
                inspect_containers(recorder, containers.stdout.split())
            except (OSError, subprocess.TimeoutExpired) as exc:
                recorder.emit('container_probe_error', detail=str(exc))
            self._llm.call = original_call
            self._parser.parse_response = original_parse
            self._session.send_keys = original_keys

    Terminus2.run = run
