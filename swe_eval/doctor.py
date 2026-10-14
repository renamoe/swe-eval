import json
import os
from pathlib import Path
import re
import socket
import tempfile
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from .io import execute, redact


def docker_access_diagnostics():
    """Collect local access evidence without treating it as daemon availability."""
    details = {'uid': os.getuid(), 'gid': os.getgid(), 'groups': os.getgroups()}
    for name in ('uid_map', 'gid_map'):
        try:
            details[name] = (Path('/proc/self') / name).read_text().strip()
        except OSError:
            pass
    try:
        details['process_restrictions'] = [line for line in Path('/proc/self/status').read_text().splitlines()
                                           if line.startswith(('NoNewPrivs:', 'Seccomp:'))]
    except OSError:
        pass
    # This probe describes the standard local socket only. Other Docker contexts
    # can point to a different endpoint; docker info remains the primary check.
    path = Path('/var/run/docker.sock')
    try:
        stat = path.stat()
        details['default_socket'] = {'uid': stat.st_uid, 'gid': stat.st_gid,
                                     'mode': oct(stat.st_mode & 0o777)}
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(2)
            connection.connect(str(path))
        details['default_socket']['connect'] = 'ok'
    except OSError as exc:
        details['default_socket_error'] = {'errno': exc.errno, 'message': str(exc)}
        if exc.errno == 1:
            details['interpretation'] = ('Socket connection denied with EPERM in this execution context. '
                                         'Check from a regular WSL terminal; this does not establish that Docker Desktop is stopped.')
    return details


def request(config, endpoint, payload=None, server_root=False):
    base = config.api_base.rstrip('/')
    if server_root and base.endswith('/v1'):
        base = base[:-3]
    url = base + endpoint
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(url, data=body, headers={
        'Authorization': 'Bearer ' + config.api_key(),
        'Content-Type': 'application/json'})
    with urlopen(req, timeout=config.api_timeout) as response:
        return json.load(response)


def check_api(config):
    if config.api_backend == 'vllm':
        models = request(config, '/models')['data']
        model = next((m for m in models if m['id'] == config.model), None)
        if model is None:
            raise ValueError('Configured model is absent from /models')
        actual_context = model.get('max_model_len')
        if actual_context is None:
            raise ValueError('/models does not expose max_model_len; cannot verify server context')
        if actual_context != config.context_length:
            raise ValueError(f'Context mismatch: client={config.context_length}, server={actual_context}')
        if not config.checkpoint or not config.model_path or not config.vllm_version or not config.chat_template:
            raise ValueError('Set EVAL_CHECKPOINT, VLLM_MODEL_PATH, VLLM_VERSION, VLLM_CHAT_TEMPLATE_FILE')
        if model.get('root') and model['root'] != config.model_path:
            raise ValueError('VLLM_MODEL_PATH differs from server /models root')
        version = request(config, '/version', server_root=True).get('version')
        if version != config.vllm_version:
            raise ValueError('VLLM_VERSION differs from server /version')
    else:
        if config.api_key() == 'EMPTY':
            raise ValueError(f'Set {config.api_key_env} in .env before calling the hosted model API')
        model, version = {'id': config.model}, None
    messages = [{'role': 'user', 'content': 'Compute 17 * 19. Return exactly one bash fenced code block containing echo and the numeric answer. Think before answering.'}]
    turns = []
    for turn in range(2):
        sample = config.sampling()
        extra = sample.pop('extra_body', {})
        payload = dict(model=config.model, messages=messages, **sample, **extra)
        response = request(config, '/chat/completions', payload)
        choice = response['choices'][0]
        message = choice['message']
        content = message.get('content')
        reasoning = message.get('reasoning_content') or message.get('reasoning')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Expected nonempty final content')
        if config.thinking and (not isinstance(reasoning, str) or not reasoning.strip()):
            raise ValueError('Expected separate nonempty reasoning and final content; check qwen3 parser')
        if '<think>' in content or '</think>' in content or message.get('tool_calls'):
            raise ValueError('Final content contains reasoning tags or native tool calls')
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Preflight response did not finish normally')
        if turn == 0:
            if len(re.findall(r'```bash\s*\n(.*?)\n```', content, re.DOTALL)) != 1:
                raise ValueError('Final content does not satisfy mini-SWE-agent text action protocol')
        else:
            action = content.strip()
            if action.startswith('```json') and action.endswith('```'):
                action = action[7:-3].strip()
            action = json.loads(action)
            if not isinstance(action.get('commands'), list) or not action['commands'] or not all(
                isinstance(c.get('keystrokes'), str) and isinstance(c.get('duration'), (int, float))
                for c in action['commands']
            ) or not isinstance(action.get('task_complete'), bool):
                raise ValueError('Final content does not satisfy Terminus-2 text action protocol')
        turns.append({'final_content': content, 'reasoning_content': reasoning, 'usage': response.get('usage')})
        # Match mini-SWE-agent: only final content becomes the next assistant turn.
        messages += [{'role': 'assistant', 'content': content},
                     {'role': 'user', 'content': 'Add 1 to the numeric result of your previous answer. Return ONLY a JSON object in exactly this format: '
                      '{"analysis":"brief explanation","plan":"echo the result","commands":[{"keystrokes":"echo 324\\n","duration":1}],"task_complete":false}. '
                      'The keystrokes value must be an echo shell command ending in a newline. duration must be a number. No markdown.'}]
    return {'model': model, 'vllm_version': version, 'turns': turns,
            'api_backend': config.api_backend, 'thinking_validated': config.thinking,
            'server_context_verified': config.api_backend == 'vllm',
            'deployment_metadata_source': 'operator supplied chat template; server version and model root checked when exposed'
            if config.api_backend == 'vllm' else 'hosted API; actual checkpoint, template and server context unavailable; temporary chain test only'}


def doctor(config, api=True, docker=True):
    checks = {}
    if docker:
        with tempfile.TemporaryDirectory() as tmp:
            for name, cmd in [('docker', ['docker', 'info']),
                              ('container', ['docker', 'run', '--rm', '--network', 'none', 'hello-world'])]:
                try:
                    result = execute(cmd, Path.cwd(), Path(tmp) / 'docker.log', timeout=120)
                    result['detail'] = (Path(tmp) / 'docker.log').read_text()[-3000:]
                    result['ok'] = result['returncode'] == 0 and not result['timed_out']
                    checks[name] = result
                except OSError as exc:
                    checks[name] = {'ok': False, 'detail': redact(str(exc))}
                if not checks[name]['ok']:
                    checks[name]['access_diagnostics'] = docker_access_diagnostics()
                    break
    if api:
        try:
            checks['api'] = {'ok': True, **check_api(config)}
        except HTTPError as exc:
            checks['api'] = {'ok': False, 'detail': redact(f'{type(exc).__name__}: {exc}'),
                             'http_status': exc.code,
                             'response_body': redact(exc.read().decode(errors='replace'))}
        except Exception as exc:
            checks['api'] = {'ok': False, 'detail': redact(f'{type(exc).__name__}: {exc}')}
    return {'ok': bool(checks) and all(c['ok'] for c in checks.values()), 'checks': checks}
