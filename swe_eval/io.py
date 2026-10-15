import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

_secrets = set()


def register_secret(value):
    if value and value != 'EMPTY':
        _secrets.add(value)


def redact(text):
    keys = {'EVAL_API_KEY', 'OPENAI_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'}
    keys.update(key for key in os.environ if key.endswith('_API_KEY'))
    for key in keys:
        register_secret(os.environ.get(key, ''))
    for secret in sorted(_secrets, key=len, reverse=True):
        text = text.replace(secret, '[REDACTED]')
    return text


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(redact(json.dumps(data, indent=2, ensure_ascii=False)) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def scrub_tree(root):
    # Upstream raw response/error logs can include credentials. Never follow symlinks.
    for path in Path(root).rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            original = path.read_text()
        except (UnicodeError, OSError):
            continue
        clean = redact(original)
        if clean != original:
            path.write_text(clean)


def _copy_lines(source, log):
    # Redact whole lines so a secret is never split across two writes.
    with source, Path(log).open('w') as target:
        for line in iter(source.readline, b''):
            target.write(redact(line.decode(errors='replace')))
            target.flush()


def _terminate(process, grace):
    """SIGTERM the whole group, then SIGKILL after grace (or on a second interrupt)."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            pass
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(command, cwd, log, env=None, timeout=None, grace=30):
    """Run an argv (never a shell) with a live redacted log; kill the group on timeout or interrupt."""
    start = time.monotonic()
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    copier = threading.Thread(target=_copy_lines, args=(process.stdout, log), daemon=True)
    copier.start()
    expired = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        expired = True
        _terminate(process, grace)
    except BaseException:
        _terminate(process, grace)
        raise
    finally:
        copier.join(timeout=10)
    return {'returncode': process.returncode, 'timed_out': expired,
            'elapsed_seconds': round(time.monotonic() - start, 3)}
