import unittest

from swe_eval.config import Config
from swe_eval.manifests import check_spec, select, spec_size

# 12 repositories with uneven sizes, like SWE-bench Verified.
GROUPS = {f'repo{r}__repo{r}-{i}': f'org/repo{r}' for r in range(12) for i in range(1 + 3 * r)}


class ManifestTests(unittest.TestCase):
    def test_seeded_sample_is_reproducible_and_seed_sensitive(self):
        spec = {'name': 's', 'sample': 10, 'seed': 7}
        first = select(GROUPS, spec)
        self.assertEqual(first, select(list(reversed(GROUPS)), spec))
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(set(first)), 10)
        self.assertNotEqual(first, select(GROUPS, dict(spec, seed=8)))

    def test_stratified_sample_covers_every_repo_once(self):
        spec = {'name': 's', 'sample': 12, 'seed': 20261015, 'stratify': 'repo'}
        chosen = select(GROUPS, spec, GROUPS)
        self.assertEqual(sorted(GROUPS[i] for i in chosen), sorted(set(GROUPS.values())))
        self.assertEqual(chosen, select(GROUPS, spec, GROUPS))
        # Larger samples still reach every group before repeating one.
        more = select(GROUPS, dict(spec, sample=15), GROUPS)
        self.assertEqual(len({GROUPS[i] for i in more}), 12)
        self.assertEqual(len(more), 15)

    def test_proportional_sample_follows_group_shares(self):
        spec = {'name': 's', 'sample': 30, 'seed': 20261015, 'stratify': 'repo-proportional'}
        chosen = select(GROUPS, spec, GROUPS)
        self.assertEqual(len(chosen), 30)
        self.assertEqual(chosen, select(list(reversed(GROUPS)), spec, GROUPS))
        counts = {}
        for task_id in chosen:
            counts[GROUPS[task_id]] = counts.get(GROUPS[task_id], 0) + 1
        sizes = {}
        for group in GROUPS.values():
            sizes[group] = sizes.get(group, 0) + 1
        # Largest remainder: every group gets floor or ceil of its exact share.
        for group, size in sizes.items():
            share = 30 * size / len(GROUPS)
            self.assertIn(counts.get(group, 0), {int(share), int(share) + 1}, group)
        self.assertNotEqual(chosen, select(GROUPS, dict(spec, seed=8), GROUPS))

    def test_explicit_and_full(self):
        self.assertEqual(select(GROUPS, {'name': 'x', 'ids': ['repo1__repo1-0', 'repo0__repo0-0']}),
                         ['repo0__repo0-0', 'repo1__repo1-0'])
        with self.assertRaisesRegex(ValueError, 'not in the dataset'):
            select(GROUPS, {'name': 'x', 'ids': ['missing']})
        self.assertEqual(select(GROUPS, {'name': 'f', 'all': True}), sorted(GROUPS))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            select(GROUPS, {'name': 's', 'sample': len(GROUPS) + 1, 'seed': 1})

    def test_spec_validation(self):
        bad = [({'name': 'x'}, 'exactly one'),
               ({'name': 'x', 'sample': 3}, 'integer seed'),
               ({'name': 'x', 'all': True, 'ids': ['a']}, 'exactly one'),
               ({'name': 'x', 'ids': ['../a']}, 'path-safe'),
               ({'name': 'x', 'ids': ['a'], 'seed': 1}, 'only applies'),
               ({'name': '../x', 'all': True}, 'safe manifest name'),
               ({'name': 'x', 'all': True, 'count': 1}, 'Unknown')]
        for spec, message in bad:
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, message):
                check_spec('swebench', spec)
        with self.assertRaisesRegex(ValueError, 'stratify'):
            check_spec('tbv2', {'name': 'x', 'sample': 2, 'seed': 1, 'stratify': 'repo'})
        with self.assertRaisesRegex(ValueError, 'stratify'):
            check_spec('swebench', {'name': 'x', 'sample': 2, 'seed': 1, 'stratify': 'org'})
        self.assertEqual(spec_size({'name': 'x', 'ids': ['a', 'b']}), 2)
        self.assertIsNone(spec_size({'name': 'x', 'all': True}))

    def test_shipped_suites(self):
        small = Config.load('qwen35-base', 'small-v1').manifests
        self.assertEqual(small['swebench'], {'name': 'small-v1', 'sample': 12, 'seed': 20261015, 'stratify': 'repo'})
        self.assertEqual(spec_size(small['tbv2']), 10)
        self.assertTrue(Config.load('qwen35-base', 'full').manifests['swebench']['all'])


if __name__ == '__main__':
    unittest.main()
