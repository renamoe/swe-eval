"""Dependency-isolated glue. Uses upstream agents and graders without overrides."""
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

from .config import Config, digest
from .io import read_json, write_json

SWE_DATASET = 'princeton-nlp/SWE-bench_Verified'


def metadata(benchmark, target):
    import importlib.util
    packages = ['mini-swe-agent', 'swebench', 'litellm'] if benchmark == 'swebench' else ['harbor', 'litellm']
    versions = {name: importlib.metadata.version(name) for name in packages}
    expected = {'mini-swe-agent': '1.17.5', 'swebench': '4.1.0', 'harbor': '0.24.0', 'litellm': '1.92.0'}
    if any(version != expected[name] for name, version in versions.items()):
        raise ValueError('Installed upstream versions differ from the supported pins; sync the locked environment')
    module = 'minisweagent' if benchmark == 'swebench' else 'harbor'
    spec = importlib.util.find_spec(module)
    root = Path(next(iter(spec.submodule_search_locations)))
    # Includes shipped prompts, parsers and agent implementation.
    sources = {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob('*'))
               if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.txt', '.j2', '.jinja', '.md')}
    write_json(target, {'python': sys.version, 'versions': versions,
                        'agent_sources_sha256': digest(sources),
                        'all_distributions': sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())})


def prepare(benchmark, profile, target):
    n = 1 if profile == 'smoke' else 10
    if benchmark == 'swebench':
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(SWE_DATASET, revision=os.getenv('EVAL_SWE_REVISION', 'main')).sha
        records = sorted(load_dataset(SWE_DATASET, split='test', revision=revision), key=lambda r: r['instance_id'])[:n]
        source = {'dataset': SWE_DATASET, 'split': 'test', 'revision': revision}
        tasks = [{'id': row['instance_id'], 'record': dict(row)} for row in records]
    else:
        from harbor.models.job.config import DatasetConfig
        dataset = DatasetConfig(name='terminal-bench', version='2.0')
        configs = asyncio.run(dataset.get_task_configs())
        configs = sorted(configs, key=lambda c: Path(c.path).name)[:n]
        tasks = []
        for item in configs:
            if not item.git_url or not item.git_commit_id or len(item.git_commit_id) != 40:
                raise ValueError('Terminal-Bench registry must resolve tasks to immutable Git commits')
            tasks.append({'id': Path(item.path).name, 'task_config': item.model_dump(mode='json')})
        source = {'dataset': 'terminal-bench', 'version': '2.0'}
    if len(tasks) != n or len({t['id'] for t in tasks}) != n:
        raise ValueError(f'Expected exactly {n} unique tasks')
    write_json(target, {'schema_version': 1, 'benchmark': benchmark, 'profile': profile,
                        'source': source, 'tasks': tasks, 'tasks_sha256': digest(tasks)})


def swe_config(config, target):
    import yaml
    from minisweagent.config import builtin_config_dir
    upstream = yaml.safe_load((builtin_config_dir / 'extra' / 'swebench.yaml').read_text())
    upstream['agent'].update(step_limit=config.max_turns, cost_limit=0)
    upstream['model'] = {'model_name': 'openai/' + config.model,
                         'model_class': 'litellm', 'cost_tracking': 'ignore_errors',
                         'model_kwargs': dict(api_base=config.api_base, timeout=config.api_timeout,
                                              num_retries=0, **config.sampling())}
    write_json(target, upstream)  # JSON is also valid YAML, no credentials embedded.


def swe_agent(config_path, task_path, output):
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models import get_model
    from minisweagent.run.extra.swebench import get_sb_environment, update_preds_file
    from minisweagent.run.utils.save import save_traj
    from .telemetry import observe_mini, inspect_containers
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    task, config = read_json(task_path), read_json(config_path)
    task_id = task['instance_id']
    trajectory = output / task_id / f'{task_id}.traj.json'
    agent = env = recorder = None
    status, submission, extra = 'Running', '', {}

    def checkpoint():
        temporary = trajectory.with_suffix('.checkpoint.tmp')
        save_traj(agent, temporary, exit_status=status, result=submission,
                  extra_info=extra, instance_id=task_id, print_path=False)
        temporary.replace(trajectory)

    try:
        model = get_model(config=config['model'])
        env = get_sb_environment(config, task)
        agent = DefaultAgent(model, env, **config['agent'])
        recorder = observe_mini(agent, output / task_id, checkpoint)
        status, submission = agent.run(task['problem_statement'])
    except BaseException as exc:
        status, submission = type(exc).__name__, ''
        extra = {'exception_str': str(exc), 'traceback': traceback.format_exc()}
    finally:
        checkpoint()
        update_preds_file(output / 'preds.json', task_id, config['model']['model_name'],
                          submission if status == 'Submitted' else '')
        if recorder:
            recorder.emit('agent_end', exit_status=status)
            inspect_containers(recorder, [env.container_id] if env.container_id else [])
        if env:
            env.cleanup()


def tb_run(config_path):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig
    from .telemetry import observe_harbor

    async def run():
        observe_harbor()
        job = await Job.create(JobConfig.model_validate(read_json(config_path)))
        await job.run()

    asyncio.run(run())


def main():
    mode, *args = sys.argv[1:]
    if mode == 'metadata':
        metadata(*args)
    elif mode == 'prepare':
        prepare(*args)
    elif mode == 'swe-config':
        swe_config(Config(**read_json(args[0])), args[1])
    elif mode == 'swe-agent':
        swe_agent(*args)
    elif mode == 'tb-run':
        tb_run(*args)
    else:
        raise ValueError(f'Unknown worker mode: {mode}')


if __name__ == '__main__':
    main()
