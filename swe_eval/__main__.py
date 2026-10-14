import argparse
import json
from pathlib import Path
import sys
from .config import Config, load_env
from .doctor import doctor
from .io import redact, write_json
from .results import compare
from .runner import prepare, run


def config_arguments(command, model_required=True):
    command.add_argument('--model', required=model_required, help='Model config name (configs/models) or TOML path')
    command.add_argument('--suite', default='smoke', help='Suite config name (configs/suites) or TOML path')
    command.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                         help='Explicit override, recorded in the snapshot (e.g. max_turns=5, sampling.temperature=0)')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Qwen local benchmark runner')
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('doctor', help='Check Docker, API and reasoning separation')
    check.add_argument('--output', type=Path)
    check.add_argument('--no-api', action='store_true', help='Docker checks only; not full acceptance')
    config_arguments(check, model_required=False)
    prep = commands.add_parser('prepare', help='Resolve, sort and freeze dataset tasks')
    prep.add_argument('benchmark', choices=['swebench', 'tbv2'])
    prep.add_argument('--profile', choices=['smoke', 'small'], default='smoke')
    for name in ('run', 'verify-scorer'):
        command = commands.add_parser(name)
        command.add_argument('benchmark', choices=['swebench', 'tbv2'])
        command.add_argument('--run-id', required=True)
        command.add_argument('--runs-dir', type=Path, default=Path('runs'))
        config_arguments(command)
    comparison = commands.add_parser('compare')
    comparison.add_argument('baseline', type=Path)
    comparison.add_argument('checkpoint', type=Path)
    comparison.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        ignored = load_env()
        if ignored:
            print('Ignoring settings in .env (only credentials are read; use configs/ or --set): '
                  + ', '.join(ignored), file=sys.stderr)
        if args.command == 'compare':
            output = args.output or args.checkpoint.parent / f'compare-{args.baseline.name}-vs-{args.checkpoint.name}'
            result = compare(args.baseline, args.checkpoint, output)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result['complete'] else 2
        if args.command == 'prepare':
            manifest = prepare(args.benchmark, args.profile)
            print(json.dumps({'benchmark': args.benchmark, 'tasks': [t['id'] for t in manifest['tasks']],
                              'tasks_sha256': manifest['tasks_sha256']}, indent=2))
            return 0
        if args.command == 'doctor':
            if not args.model and not args.no_api:
                raise ValueError('doctor needs --model unless --no-api is given')
            config = Config.load(args.model, args.suite, args.set) if args.model else None
            result = doctor(config, api=not args.no_api)
            if args.output:
                write_json(args.output, result)
            print(redact(json.dumps(result, indent=2, ensure_ascii=False)))
            return 0 if result['ok'] else 2
        config = Config.load(args.model, args.suite, args.set)
        output, result = run(args.benchmark, args.run_id, config,
                             args.runs_dir, oracle=args.command == 'verify-scorer')
        print(json.dumps({'output': str(output), **result}, indent=2, ensure_ascii=False))
        if args.command == 'verify-scorer' and result['success_count'] != result['task_count']:
            return 2
        return 0 if result['complete'] else 2
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(redact(f'{type(exc).__name__}: {exc}'), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
