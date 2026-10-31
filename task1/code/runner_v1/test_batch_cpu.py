import unittest
import numpy as np

from task1_batch_benchmark import choose_batch, select_ids


class BatchTests(unittest.TestCase):
    def test_fastest_eligible_batch_excludes_oom_and_memory_pressure(self):
        rows = [
            {"batch_size": 8, "mean_seconds": 5., "projected_peak_bytes": 20},
            {"batch_size": 16, "mean_seconds": 3., "projected_peak_bytes": 70},
            {"batch_size": 32, "mean_seconds": 2., "projected_peak_bytes": 90},
            {"batch_size": 64, "oom": True},
        ]
        self.assertEqual(choose_batch(rows, 100), 16)

    def test_ties_choose_smaller_batch(self):
        rows = [{"batch_size": size, "mean_seconds": 3., "projected_peak_bytes": 10}
                for size in [16, 8]]
        self.assertEqual(choose_batch(rows, 100), 8)

    def test_length_and_seed_selection_preserves_longest_without_duplicates(self):
        texts = ["x" * size for size in range(20)]
        first = select_ids(texts, 8, 2, np.random.default_rng(6304))
        second = select_ids(texts, 8, 2, np.random.default_rng(6304))
        self.assertEqual(first, second)
        self.assertEqual(first[:2], [19, 18])
        self.assertEqual(len(set(first)), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
