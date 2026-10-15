import argparse
import json
from pathlib import Path
import signal
import sys
import time
from .config import Config, load_env, load_suite
from .doctor import doctor
from .io import redact, write_json
from .report import report
from .runner import prepare, regrade, run_suite


def interrupt(signum, frame):
    raise KeyboardInterrupt


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
    prep = commands.add_parser('prepare', help="Resolve and freeze a suite's task lists under profiles/")
    prep.add_argument('benchmark', nargs='?', choices=['swebench', 'tbv2'],
                      help='Only this benchmark (default: every benchmark in the suite)')
    prep.add_argument('--suite', default='smoke', help='Suite config name (configs/suites) or TOML path')
    for name in ('run', 'verify-scorer'):
        command = commands.add_parser(name, help='Run the suite in the foreground; Ctrl-C stops and cleans up')
        command.add_argument('benchmark', nargs='?', choices=['swebench', 'tbv2'],
                             help='Only this benchmark (default: every benchmark in the suite)')
        command.add_argument('--run-id', help='Default <model>__<suite>__<benchmark>__<timestamp>')
        command.add_argument('--runs-dir', type=Path, default=Path('runs'))
        config_arguments(command)
    summary = commands.add_parser('report', help='Compare runs per benchmark; the first model is the baseline')
    summary.add_argument('runs', nargs='+', help='Model config names (latest complete run, needs --suite) or run directories')
    summary.add_argument('--suite', help='Suite config name used to find runs by model')
    summary.add_argument('--benchmark', action='append', choices=['swebench', 'tbv2'],
                         help='Only this benchmark (repeatable; default: the suite benchmarks)')
    summary.add_argument('--runs-dir', type=Path, default=Path('runs'))
    summary.add_argument('--output', type=Path, help='Default <runs-dir>/reports/<timestamp>')
    comparison = commands.add_parser('compare', help='Two-run report (alias of report)')
    comparison.add_argument('baseline', type=Path)
    comparison.add_argument('checkpoint', type=Path)
    comparison.add_argument('--output', type=Path)
    again = commands.add_parser('regrade', help='Score a finished SWE run under another policy into a new run')
    again.add_argument('run', type=Path, help='Run directory')
    again.add_argument('--policy', default='final-workspace', choices=['final-workspace'])
    again.add_argument('--run-id', help='Default <source run-id>__<policy>')
    args = parser.parse_args(argv)
    try:
        ignored = load_env()
        if ignored:
            print('Ignoring settings in .env (only credentials are read; use configs/ or --set): '
                  + ', '.join(ignored), file=sys.stderr)
        if args.command in ('report', 'compare'):
            if args.command == 'compare':
                refs, suite, benchmarks = [args.baseline, args.checkpoint], None, ()
                output = args.output or args.checkpoint.parent / f'compare-{args.baseline.name}-vs-{args.checkpoint.name}'
            else:
                refs, suite = args.runs, args.suite
                benchmarks = args.benchmark or (load_suite(suite)[1]['benchmarks'] if suite else ())
                output = args.output or args.runs_dir / 'reports' / time.strftime('%Y%m%d-%H%M%S')
            result = report(refs, output, suite, benchmarks, getattr(args, 'runs_dir', Path('runs')),
                            single=args.command == 'compare')
            print(f'Report: {output / "report.html"}')
            print(json.dumps([{'benchmark': b['benchmark'], 'differences': [d['key'] for d in b['differences']],
                               'models': [{k: m[k] for k in ('label', 'complete', 'passed_count', 'scored_count',
                                                             'task_count', 'success_rate', 'success_rate_delta',
                                                             'gained', 'regressed')} for m in b['models']]}
                              for b in result['benchmarks']], indent=2, ensure_ascii=False))
            return 0 if all(m['complete'] for b in result['benchmarks'] for m in b['models']) else 2
        if args.command == 'regrade':
            signal.signal(signal.SIGTERM, interrupt)
            signal.signal(signal.SIGINT, signal.default_int_handler)
            output, summary = regrade(args.run, args.policy, run_id=args.run_id)
            print(json.dumps({'output': str(output), **{k: summary[k] for k in (
                'complete', 'stopped', 'success_count', 'task_count', 'success_rate', 'regraded_count')}},
                indent=2, ensure_ascii=False))
            return 130 if summary['stopped'] else 0 if summary['complete'] else 2
        if args.command == 'prepare':
            name, suite = load_suite(args.suite)
            output = []
            for benchmark in [args.benchmark] if args.benchmark else suite['benchmarks']:
                if benchmark not in suite['manifests']:
                    raise ValueError(f'Suite {name} does not include {benchmark}')
                manifest = prepare(benchmark, suite['manifests'][benchmark])
                output.append({'benchmark': benchmark, 'manifest': manifest['name'],
                               'tasks': [t['id'] for t in manifest['tasks']],
                               'tasks_sha256': manifest['tasks_sha256']})
            print(json.dumps(output, indent=2))
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
        # SIGTERM stops the run exactly like Ctrl-C: terminate, clean up, write the summary.
        # Restore SIGINT too: shells start background jobs with SIGINT ignored.
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        results = run_suite(config, args.runs_dir, args.run_id, [args.benchmark] if args.benchmark else None,
                            oracle=args.command == 'verify-scorer')
        print(json.dumps([{k: r[k] for k in ('benchmark', 'output', 'complete', 'stopped', 'success_count',
                                              'task_count', 'success_rate', 'status_counts', 'fatal_error')}
                          for r in results], indent=2, ensure_ascii=False))
        if any(r['stopped'] for r in results):
            return 130
        if args.command == 'verify-scorer' and any(r['success_count'] != r['task_count'] for r in results):
            return 2
        return 0 if all(r['complete'] for r in results) else 2
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(redact(f'{type(exc).__name__}: {exc}'), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
