"""Suite task-list specs: explicit IDs, a seeded sample (SWE may stratify by repo, round-robin or
proportional), or the full split.

Stdlib only; imported by the main CLI and by both isolated worker environments.
"""
import hashlib
import re

NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
SAFE_ID = re.compile(r'[A-Za-z0-9_.-]+')
SPEC_FIELDS = {'name', 'ids', 'sample', 'seed', 'stratify', 'all'}
# repo: round-robin, every repo once before any twice. repo-proportional: per-repo counts follow dataset shares.
STRATIFY = {'repo', 'repo-proportional'}


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def safe_id(value):
    return isinstance(value, str) and SAFE_ID.fullmatch(value) and value not in ('.', '..')


def check_spec(benchmark, spec):
    label = f'manifests.{benchmark}'
    if not isinstance(spec, dict):
        raise ValueError(f'{label} must be a table')
    unknown = set(spec) - SPEC_FIELDS
    if unknown:
        raise ValueError(f'Unknown {label} fields: ' + ', '.join(sorted(unknown)))
    if not isinstance(spec.get('name'), str) or not NAME.fullmatch(spec['name']):
        raise ValueError(f'{label}.name must be a safe manifest name')
    if sum(k in spec for k in ('ids', 'sample', 'all')) != 1:
        raise ValueError(f'{label} needs exactly one of ids, sample or all')
    if 'ids' in spec:
        ids = spec['ids']
        if not isinstance(ids, list) or not ids or not all(safe_id(i) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'{label}.ids must be a non-empty list of unique, path-safe task IDs')
    if 'sample' in spec:
        if not is_int(spec['sample']) or spec['sample'] <= 0 or not is_int(spec.get('seed')):
            raise ValueError(f'{label}.sample needs a positive count and an integer seed')
    elif 'seed' in spec:
        raise ValueError(f'{label}.seed only applies to sample')
    if 'stratify' in spec and (spec['stratify'] not in STRATIFY or benchmark != 'swebench' or 'sample' not in spec):
        raise ValueError(f'{label}.stratify supports only "repo" or "repo-proportional" for a swebench sample')
    if 'all' in spec and spec['all'] is not True:
        raise ValueError(f'{label}.all must be true')
    return dict(spec)


def spec_size(spec):
    """Planned task count, or None for the full split (known after preparation)."""
    return len(spec['ids']) if 'ids' in spec else spec.get('sample')


def rank(seed, key):
    # A hash order, not a PRNG: identical across Python versions and environments.
    return hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()


def select(ids, spec, groups=None):
    """Return the sorted task IDs a spec selects from the candidate IDs."""
    ids = sorted(set(ids))
    if 'all' in spec:
        return ids
    if 'ids' in spec:
        missing = sorted(set(spec['ids']) - set(ids))
        if missing:
            raise ValueError('Manifest IDs not in the dataset: ' + ', '.join(missing))
        return sorted(spec['ids'])
    n, seed = spec['sample'], spec['seed']
    if n > len(ids):
        raise ValueError(f'Sample of {n} exceeds the {len(ids)} available tasks')
    if not spec.get('stratify'):
        return sorted(sorted(ids, key=lambda i: rank(seed, i))[:n])
    buckets = {}
    for task_id in ids:
        buckets.setdefault(groups[task_id], []).append(task_id)
    # Round-robin over groups in seeded order: every group gets one task before any gets two.
    queues = [sorted(members, key=lambda i: rank(seed, i))
              for _, members in sorted(buckets.items(), key=lambda item: rank(seed, item[0]))]
    if spec['stratify'] == 'repo-proportional':
        # Largest-remainder allocation of n by group share; equal remainders go to groups in seeded order.
        quotas = [n * len(q) // len(ids) for q in queues]
        spare = sorted(range(len(queues)), key=lambda g: -(n * len(queues[g]) % len(ids)))
        for g in spare[:n - sum(quotas)]:
            quotas[g] += 1
        return sorted(task_id for q, k in zip(queues, quotas) for task_id in q[:k])
    chosen, depth = [], 0
    while len(chosen) < n:
        chosen.extend(q[depth] for q in queues if depth < len(q))
        depth += 1
    # The last round may overshoot; keep its first groups in seeded order.
    return sorted(chosen[:n])
