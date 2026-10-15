from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import urlsplit

from .manifests import check_spec

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / 'configs'
BENCHMARKS = ('swebench', 'tbv2')
ENV_PASSTHROUGH = {'EVAL_SWE_REVISION'}
LEGACY_ENV = {'VLLM_MODEL_PATH', 'VLLM_VERSION', 'VLLM_CHAT_TEMPLATE_FILE'}

# name: (type, default); default None means required.
MODEL_FIELDS = {
    'model': (str, None), 'base_url': (str, None), 'context_length': (int, None),
    'server': (str, 'other'), 'checkpoint': (str, ''), 'api_key_env': (str, ''),
    'reasoning': (bool, False), 'model_path': (str, ''), 'vllm_version': (str, ''),
    'chat_template': (str, ''), 'sampling': (dict, None), 'extra_body': (dict, {}),
}
SAMPLING_FIELDS = {'temperature': float, 'top_p': float, 'presence_penalty': float, 'frequency_penalty': float}
SUITE_FIELDS = {
    'benchmarks': (list, None), 'manifests': (dict, None), 'max_tokens': (int, None),
    'max_turns': (int, None), 'swe_agent_timeout': (int, None), 'swe_score_timeout': (int, None),
    'api_timeout': (int, None), 'swe_scoring': (str, 'submitted-only'),
    'record_terminal_session': (bool, False), 'concurrency': (int, 1),
}
# extra_body is merged into the request body; it must not replace core request fields.
RESERVED_BODY = {'model', 'messages', 'stream', 'tools', 'max_tokens', *SAMPLING_FIELDS}


def load_env(path=Path('.env')):
    """Read literal KEY=VALUE entries. No interpolation or shell execution.

    Only credentials and tool environment belong here. Returns legacy EVAL_*/VLLM_*
    keys that were skipped; evaluation settings live in configs/.
    """
    ignored = []
    if not path.exists():
        return ignored
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        key = key.strip()
        if not sep or not key.replace('_', '').isalnum():
            raise ValueError(f'Invalid .env entry at line {number}')
        if (key.startswith('EVAL_') and key not in ENV_PASSTHROUGH) or key in LEGACY_ENV:
            ignored.append(key)
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return ignored


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def config_path(kind, ref):
    """Resolve a config name (configs/<kind>/<name>.toml) or an explicit .toml path."""
    if ref.endswith('.toml') or '/' in ref or os.sep in ref:
        path = Path(ref)
    elif re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', ref):
        path = CONFIGS / kind / f'{ref}.toml'
    else:
        raise ValueError(f'Invalid {kind[:-1]} name: {ref}')
    if not path.is_file():
        raise ValueError(f'{kind[:-1].capitalize()} config not found: {path}')
    return path


def read_toml(path):
    try:
        return tomllib.loads(Path(path).read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f'Invalid TOML in {path}: {exc}') from None


def merge(base, child):
    result = deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_model(ref):
    path = config_path('models', ref)
    data = read_toml(path)
    parent = data.pop('extends', None)
    if parent is not None:
        if not isinstance(parent, str):
            raise ValueError(f'{path}: extends must be a model config name')
        base = read_toml(config_path('models', parent))
        if 'extends' in base:
            raise ValueError(f'{path}: extends supports a single level; {parent} also extends')
        data = merge(base, data)
    return path.stem, data


def load_suite(ref):
    path = config_path('suites', ref)
    return path.stem, check_suite(read_toml(path))


def parse_overrides(items):
    """Parse --set key=value. Values use TOML syntax; bare words are strings."""
    overrides = {}
    for item in items or ():
        key, sep, raw = item.partition('=')
        key = key.strip()
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*', key):
            raise ValueError(f'--set expects key=value, got {item!r}')
        try:
            value = tomllib.loads(f'v = {raw.strip()}')['v']
        except tomllib.TOMLDecodeError:
            value = raw.strip()
        overrides[key] = value
    return overrides


def apply_overrides(model, suite, overrides):
    model, suite = deepcopy(model), deepcopy(suite)
    for key, value in overrides.items():
        top, *rest = key.split('.')
        target = model if top in MODEL_FIELDS else suite if top in SUITE_FIELDS else None
        if target is None:
            raise ValueError(f'Unknown --set field: {key}')
        for part in [top, *rest][:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f'--set {key}: {part} is not a table')
        target[[top, *rest][-1]] = value
    return model, suite


def check_fields(data, schema, label):
    unknown = set(data) - set(schema)
    if unknown:
        raise ValueError(f'Unknown {label} fields (credentials belong in environment variables): '
                         + ', '.join(sorted(unknown)))
    result = {}
    for name, (kind, default) in schema.items():
        if name not in data:
            if default is None:
                raise ValueError(f'Missing required {label} field: {name}')
            result[name] = deepcopy(default)
            continue
        value = data[name]
        if kind is int and (isinstance(value, bool) or not isinstance(value, int)) or \
                kind is not int and not isinstance(value, kind):
            raise ValueError(f'Invalid type for {label} field {name}: expected {kind.__name__}')
        result[name] = value
    return result


def check_suite(data):
    suite = check_fields(data, SUITE_FIELDS, 'suite')
    if not suite['benchmarks'] or any(b not in BENCHMARKS for b in suite['benchmarks']) \
            or len(set(suite['benchmarks'])) != len(suite['benchmarks']):
        raise ValueError(f'Suite benchmarks must be a non-empty unique subset of {", ".join(BENCHMARKS)}')
    if set(suite['manifests']) != set(suite['benchmarks']):
        raise ValueError('Suite manifests must define one manifest per benchmark')
    suite['manifests'] = {b: check_spec(b, spec) for b, spec in suite['manifests'].items()}
    for name in ('max_tokens', 'max_turns', 'swe_agent_timeout', 'swe_score_timeout', 'api_timeout'):
        if suite[name] <= 0:
            raise ValueError(f'{name} must be positive')
    if suite['swe_scoring'] not in ('submitted-only', 'final-workspace'):
        raise ValueError('swe_scoring must be submitted-only or final-workspace')
    if not 1 <= suite['concurrency'] <= 4:
        raise ValueError('concurrency must be between 1 and 4')
    return suite


def check_model(data):
    model = check_fields(data, MODEL_FIELDS, 'model')
    if model['server'] not in ('vllm', 'other'):
        raise ValueError('server must be vllm or other')
    if model['api_key_env'] and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', model['api_key_env']):
        raise ValueError('api_key_env must name an environment variable')
    url = urlsplit(model['base_url'])
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('base_url must be an http(s) URL without credentials/query/fragment')
    if model['context_length'] <= 0:
        raise ValueError('context_length must be positive')
    sampling = dict(model['sampling'])
    unknown = set(sampling) - set(SAMPLING_FIELDS)
    if unknown:
        raise ValueError('Unknown sampling fields (use extra_body for server-specific parameters): '
                         + ', '.join(sorted(unknown)))
    if 'temperature' not in sampling:
        raise ValueError('sampling.temperature is required')
    for name, value in sampling.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'sampling.{name} must be a finite number')
        sampling[name] = float(value)
    if not 0 <= sampling['temperature'] <= 2 or not 0 < sampling.get('top_p', 1) <= 1 or \
            not all(-2 <= sampling.get(n, 0) <= 2 for n in ('presence_penalty', 'frequency_penalty')):
        raise ValueError('Invalid sampling parameters')
    model['sampling'] = sampling
    reserved = RESERVED_BODY & set(model['extra_body'])
    if reserved:
        raise ValueError('extra_body must not set core request fields: ' + ', '.join(sorted(reserved)))
    return model


def template_sha256(path):
    if not path:
        return ''
    file = Path(path) if Path(path).is_absolute() else ROOT / path
    if not file.is_file():
        raise ValueError(f'chat_template not found: {path}')
    return hashlib.sha256(file.read_bytes()).hexdigest()


def base_env():
    """Child-process environment without model credentials."""
    env = {k: v for k, v in os.environ.items() if k != 'OPENAI_API_KEY'}
    env['MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT'] = '1'
    env['LITELLM_LOG'] = 'ERROR'
    env['HF_HUB_DISABLE_TELEMETRY'] = '1'
    env['PYTHONPATH'] = str(ROOT)
    return env


@dataclass(frozen=True)
class Config:
    """Resolved model x suite settings. Built from configs/ at submit time, then only from snapshots."""
    model_config: str
    suite_config: str
    model: str
    base_url: str
    context_length: int
    sampling_params: dict
    benchmarks: tuple
    manifests: dict
    max_tokens: int
    max_turns: int
    swe_agent_timeout: int
    swe_score_timeout: int
    api_timeout: int
    server: str = 'other'
    checkpoint: str = ''
    api_key_env: str = ''
    reasoning: bool = False
    model_path: str = ''
    vllm_version: str = ''
    chat_template: str = ''
    chat_template_sha256: str = ''
    extra_body: dict = None
    swe_scoring: str = 'submitted-only'
    record_terminal_session: bool = False
    concurrency: int = 1
    overrides: dict = None

    @classmethod
    def load(cls, model, suite, overrides=None):
        model_name, model_data = load_model(model)
        suite_name, suite_data = load_suite(suite)
        overrides = parse_overrides(overrides) if isinstance(overrides, (list, tuple)) else dict(overrides or {})
        model_data, suite_data = apply_overrides(model_data, suite_data, overrides)
        return cls.build(model_name, check_model(model_data), suite_name, check_suite(suite_data), overrides)

    @classmethod
    def build(cls, model_name, model, suite_name, suite, overrides):
        model, suite = dict(model), dict(suite)
        if suite['max_tokens'] >= model['context_length']:
            raise ValueError(f"max_tokens ({suite['max_tokens']}) must be less than context_length ({model['context_length']})")
        sampling = model.pop('sampling')
        return cls(model_config=model_name, suite_config=suite_name, sampling_params=sampling,
                   chat_template_sha256=template_sha256(model['chat_template']),
                   overrides=overrides, **model, **{**suite, 'benchmarks': tuple(suite['benchmarks'])})

    def snapshot(self):
        """Frozen, credential-free record written to runs/<id>/config.json."""
        values = asdict(self)
        model = {f: values[f] for f in MODEL_FIELDS if f != 'sampling'}
        model['sampling'] = values['sampling_params']
        suite = {f: values[f] for f in SUITE_FIELDS}
        suite['benchmarks'] = list(suite['benchmarks'])
        return {'schema_version': 3, 'model_config': self.model_config, 'model': model,
                'suite_config': self.suite_config, 'suite': suite,
                'chat_template_sha256': self.chat_template_sha256, 'overrides': self.overrides}

    @classmethod
    def from_snapshot(cls, data):
        if data.get('schema_version') != 3:
            raise ValueError('Unsupported config snapshot; expected schema_version 3')
        result = cls.build(data['model_config'], check_model(data['model']),
                           data['suite_config'], check_suite(data['suite']), data['overrides'])
        if data['chat_template_sha256'] != result.chat_template_sha256:
            raise ValueError('chat_template changed since the run was submitted')
        return result

    def comparison_settings(self):
        """Settings that affect outcomes; identity and endpoint fields excluded."""
        snapshot = self.snapshot()
        model = {k: v for k, v in snapshot['model'].items()
                 if k not in ('model', 'base_url', 'checkpoint', 'api_key_env', 'model_path')}
        # Concurrency is recorded once, at the top level of comparison.json.
        suite = {k: v for k, v in snapshot['suite'].items() if k != 'concurrency'}
        return {'model': model, 'suite': suite, 'chat_template_sha256': self.chat_template_sha256}

    def sampling(self):
        parameters = dict(self.sampling_params, max_tokens=self.max_tokens)
        if self.extra_body:
            parameters['extra_body'] = deepcopy(self.extra_body)
        return parameters

    def api_key(self):
        """Read only api_key_env. Empty api_key_env declares an unauthenticated endpoint."""
        from .io import register_secret
        if not self.api_key_env:
            return 'EMPTY'
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f'{self.api_key_env} is not set (required by model config {self.model_config})')
        register_secret(key)
        return key

    def child_env(self, api=True):
        env = base_env()
        if api:
            env['OPENAI_API_KEY'] = self.api_key()
            env['OPENAI_API_BASE'] = self.base_url
        return env
