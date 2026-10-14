from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent


def load_env(path=Path('.env')):
    """Read literal KEY=VALUE entries. No interpolation or shell execution."""
    if not path.exists():
        return
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        if not sep or not key.strip().replace('_', '').isalnum():
            raise ValueError(f'Invalid .env entry at line {number}')
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    model_instance: str = ''
    api_backend: str = 'vllm'
    api_key_env: str = 'EVAL_API_KEY'
    thinking: bool = True
    record_terminal_session: bool = True
    api_base: str = 'http://localhost:8000/v1'
    model: str = 'qwen35-9b'
    checkpoint: str = ''
    model_path: str = ''
    vllm_version: str = ''
    chat_template: str = ''
    context_length: int = 131072
    max_tokens: int = 16384
    max_turns: int = 100
    swe_agent_timeout: int = 3600
    swe_score_timeout: int = 1800
    api_timeout: int = 180
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 0.0

    @classmethod
    def from_env(cls, instance=None):
        values = {}
        defaults = cls()
        instance = instance or os.environ.get('EVAL_MODEL_INSTANCE')
        if instance:
            path = Path(instance)
            if not path.is_file():
                if not instance.replace('-', '').replace('_', '').replace('.', '').isalnum():
                    raise ValueError('Model instance must be a JSON path or a preset name')
                path = ROOT / 'models' / f'{instance}.json'
            preset = json.loads(path.read_text())
            unknown = set(preset) - set(asdict(defaults))
            if unknown:
                raise ValueError('Unknown model instance fields (credentials belong in environment variables): ' + ', '.join(sorted(unknown)))
            for name, value in preset.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and type(getattr(defaults, name)) is float:
                    preset[name] = float(value)
                    continue
                if type(value) is not type(getattr(defaults, name)):
                    raise ValueError(f'Invalid type for model instance field {name}')
            values.update(preset)
            values['model_instance'] = preset.get('model_instance') or path.stem
        for name in asdict(defaults):
            if name == 'model_instance':
                continue
            key = {'model_path': 'VLLM_MODEL_PATH', 'vllm_version': 'VLLM_VERSION',
                   'chat_template': 'VLLM_CHAT_TEMPLATE_FILE'}.get(name, 'EVAL_' + name.upper())
            if os.environ.get(key):
                value = os.environ[key]
                if name == 'chat_template':
                    value = Path(value).read_text()
                elif type(getattr(defaults, name)) is bool:
                    if value.lower() not in ('true', 'false'):
                        raise ValueError(f'{key} must be true or false')
                    value = value.lower() == 'true'
                else:
                    value = type(getattr(defaults, name))(value)
                values[name] = value
        result = cls(**values)
        if result.api_backend not in ('vllm', 'openai-compatible'):
            raise ValueError('EVAL_API_BACKEND must be vllm or openai-compatible')
        if result.api_backend == 'openai-compatible' and result.thinking:
            raise ValueError('Temporary OpenAI-compatible mode requires thinking=false; use vllm for Qwen reasoning validation')
        if not result.api_key_env or not result.api_key_env.replace('_', '').isalnum():
            raise ValueError('api_key_env must name an environment variable')
        url = urlsplit(result.api_base)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('EVAL_API_BASE must be an http(s) URL without credentials/query/fragment')
        for name in ('context_length', 'max_tokens', 'max_turns', 'swe_agent_timeout', 'swe_score_timeout', 'api_timeout', 'top_k'):
            if getattr(result, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if result.max_tokens >= result.context_length:
            raise ValueError('max_tokens must be less than context_length')
        if not all(math.isfinite(getattr(result, n)) for n in ('temperature', 'top_p', 'presence_penalty')):
            raise ValueError('Sampling parameters must be finite')
        if not 0 <= result.temperature <= 2 or not 0 < result.top_p <= 1 or not -2 <= result.presence_penalty <= 2:
            raise ValueError('Invalid sampling parameters')
        return result

    def snapshot(self):
        return asdict(self)

    def sampling(self):
        parameters = dict(temperature=self.temperature, top_p=self.top_p,
                          presence_penalty=self.presence_penalty, max_tokens=self.max_tokens)
        if self.api_backend == 'vllm':
            parameters['extra_body'] = {'top_k': self.top_k,
                                        'chat_template_kwargs': {'enable_thinking': self.thinking}}
        return parameters

    def api_key(self):
        from .io import register_secret
        key = os.environ.get('EVAL_API_KEY') or os.environ.get(self.api_key_env) or 'EMPTY'
        register_secret(key)
        return key

    def child_env(self):
        env = dict(os.environ)
        env['OPENAI_API_KEY'] = self.api_key()
        env['OPENAI_API_BASE'] = self.api_base
        env['MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT'] = '1'
        env['LITELLM_LOG'] = 'ERROR'
        env['HF_HUB_DISABLE_TELEMETRY'] = '1'
        env['PYTHONPATH'] = str(ROOT)
        return env
