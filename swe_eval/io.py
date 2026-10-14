import json
import os
from pathlib import Path
import signal
import subprocess
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


def execute(command, cwd, log, env=None, timeout=None):
    """Run an argv (never a shell); kill the process group on timeout or interrupt."""
    start = time.monotonic()
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    # Spool output outside the experiment; copy only redacted output into artifacts.
    import tempfile
    with tempfile.TemporaryFile(mode='w+b') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        expired = False
        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            expired = isinstance(exc, subprocess.TimeoutExpired)
            if not expired:
                raise
        finally:
            stream.seek(0)
            log.write_text(redact(stream.read().decode(errors='replace')))
    return {'returncode': process.returncode, 'timed_out': expired,
            'elapsed_seconds': round(time.monotonic() - start, 3)}
