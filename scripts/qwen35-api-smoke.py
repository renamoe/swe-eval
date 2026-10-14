"""Run with environments/swebench/.venv/bin/python from the project root."""
import json
from pathlib import Path
import sys

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swe_eval.config import Config, load_env

load_env()
config = Config.load('qwen35-base', 'smoke')
client = OpenAI(base_url=config.base_url, api_key=config.api_key(),
                timeout=config.api_timeout, max_retries=0)
models = client.models.list()
assert config.model in [model.id for model in models.data]
response = client.chat.completions.create(
    model=config.model,
    messages=[{'role': 'user', 'content': 'What is 1 + 1? Answer with just the number.'}],
    temperature=0, max_tokens=64,
    extra_body={'chat_template_kwargs': {'enable_thinking': False}},
)
output = Path('runs/qwen35-api-minimal')
output.mkdir(parents=True, exist_ok=True)
(output / 'openai-smoke.json').write_text(response.model_dump_json(indent=2) + '\n')
assert response.choices[0].finish_reason == 'stop', response.choices[0]
assert response.choices[0].message.content.strip() == '2', response.choices[0]
print(json.dumps({'ok': True, 'model': response.model, 'answer': '2',
                  'usage': response.usage.model_dump()}, indent=2))
