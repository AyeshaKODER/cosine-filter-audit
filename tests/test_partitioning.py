import unittest

from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
import numpy as np

from cosine_filter_audit.task import create_partitioner


class PartitioningTests(unittest.TestCase):
    def test_creates_seeded_dirichlet_label_partitioner(self):
        partitioner = create_partitioner("dirichlet", 10, 0.3, 17)

        self.assertIsInstance(partitioner, DirichletPartitioner)
        np.testing.assert_array_equal(partitioner._alpha, [0.3] * 10)
        self.assertEqual(partitioner._seed, 17)

    def test_creates_iid_partitioner(self):
        partitioner = create_partitioner("iid", 10)

        self.assertIsInstance(partitioner, IidPartitioner)

    def test_rejects_invalid_partition_settings(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            create_partitioner("dirichlet", 10, 0.0, 17)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            create_partitioner("unknown", 10)


if __name__ == "__main__":
    unittest.main()
