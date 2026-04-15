"""
Parity tests for netexprs.expand_batch — the new Rust function for
parallelizing expansion across multiple seed sets.

Tests at two levels:
  1. Low-level: call netexprs.expand_batch directly and compare each row
     against netexprs.expand (single-seed Rust expansion).
  2. High-level: call _run_expansions_rust (batch) and compare against
     _run_expansions_python (loop) through the GlobalMetabolicNetwork API.
"""

import unittest
import numpy as np
from random import sample, seed as random_seed

import networkExpansionPy.lib as ne

try:
    import netexprs
except ImportError:
    netexprs = None


# ---------------------------------------------------------------------------
# Low-level tests: netexprs.expand_batch vs netexprs.expand
# ---------------------------------------------------------------------------

@unittest.skipIf(netexprs is None, "netexprs not installed")
class TestExpandBatchLowLevel(unittest.TestCase):
    """Direct comparison of expand_batch rows against single expand calls."""

    @classmethod
    def setUpClass(cls):
        cls.kegg = ne.GlobalMetabolicNetwork()
        cls.kegg.pruneInconsistentReactions()
        cls.kegg.convertToIrreversible()
        cls.kegg._ensure_rust_ready()

        seeds_df = ne.pd.read_csv(ne.asset_path + "/compounds/seeds.Goldford2022.csv")
        cls.all_seeds = list(set(seeds_df["ID"]))

        ra = cls.kegg._rust_arrays
        cls.ra = ra

    def _run_single(self, x0):
        """Run a single expansion via netexprs.expand."""
        ra = self.ra
        return netexprs.expand(
            ra["rt_data"], ra["rt_indices"], ra["rt_indptr"], ra["n_reactions"],
            ra["p_data"], ra["p_indices"], ra["p_indptr"], ra["n_compounds"],
            x0, ra["b"],
        )

    def _run_batch(self, x_batch):
        """Run batch expansion via netexprs.expand_batch."""
        ra = self.ra
        return netexprs.expand_batch(
            ra["rt_data"], ra["rt_indices"], ra["rt_indptr"], ra["n_reactions"],
            ra["p_data"], ra["p_indices"], ra["p_indptr"], ra["n_compounds"],
            x_batch, ra["b"],
        )

    def test_single_seed_set_matches(self):
        """Batch with one row should equal single expand."""
        x0 = self.kegg.initialize_metabolite_vector(self.all_seeds).astype(np.uint8)
        x_single, y_single = self._run_single(x0)

        x_batch_in = x0.reshape(1, -1)
        x_batch_out, y_batch_out = self._run_batch(x_batch_in)

        np.testing.assert_array_equal(x_single, x_batch_out[0])
        np.testing.assert_array_equal(y_single, y_batch_out[0])

    def test_multiple_seed_sets(self):
        """Batch with N rows should equal N individual expand calls."""
        random_seed(42)
        seed_sets = [
            self.all_seeds,
            self.all_seeds[:10],
            sample(self.all_seeds, 20),
            sample(self.all_seeds, 5),
            ["C00001"],  # water only
        ]

        n_compounds = self.ra["n_compounds"]
        x_batch_in = np.zeros((len(seed_sets), n_compounds), dtype=np.uint8)
        for i, seeds in enumerate(seed_sets):
            x_batch_in[i] = self.kegg.initialize_metabolite_vector(seeds).astype(np.uint8)

        x_batch_out, y_batch_out = self._run_batch(x_batch_in)

        for i in range(len(seed_sets)):
            x_single, y_single = self._run_single(x_batch_in[i])
            np.testing.assert_array_equal(
                x_single, x_batch_out[i],
                err_msg=f"Compound mismatch for seed set {i}",
            )
            np.testing.assert_array_equal(
                y_single, y_batch_out[i],
                err_msg=f"Reaction mismatch for seed set {i}",
            )

    def test_empty_seed_set(self):
        """A row of all zeros (no seeds) should produce no expansion."""
        n_compounds = self.ra["n_compounds"]
        x_batch_in = np.zeros((1, n_compounds), dtype=np.uint8)
        x_batch_out, y_batch_out = self._run_batch(x_batch_in)

        np.testing.assert_array_equal(x_batch_out[0], np.zeros(n_compounds, dtype=np.uint8))
        np.testing.assert_array_equal(y_batch_out[0], np.zeros(self.ra["n_reactions"], dtype=np.uint8))

    def test_output_shapes(self):
        """Output arrays should have correct shapes."""
        n_seeds = 7
        n_compounds = self.ra["n_compounds"]
        n_reactions = self.ra["n_reactions"]

        x_batch_in = np.zeros((n_seeds, n_compounds), dtype=np.uint8)
        for i in range(n_seeds):
            subset = sample(self.all_seeds, min(i + 1, len(self.all_seeds)))
            x_batch_in[i] = self.kegg.initialize_metabolite_vector(subset).astype(np.uint8)

        x_out, y_out = self._run_batch(x_batch_in)

        self.assertEqual(x_out.shape, (n_seeds, n_compounds))
        self.assertEqual(y_out.shape, (n_seeds, n_reactions))


# ---------------------------------------------------------------------------
# Low-level toy network tests
# ---------------------------------------------------------------------------

@unittest.skipIf(netexprs is None, "netexprs not installed")
class TestExpandBatchLowLevelToy(unittest.TestCase):
    """Low-level batch tests on a small toy network for easy debugging."""

    @classmethod
    def setUpClass(cls):
        toy = ne.GlobalMetabolicNetwork("dev")
        rxns = [
            (["A", "B"], ["C"]),
            (["C", "D"], ["E", "F"]),
            (["E", "F"], ["G"]),
            (["G", "H"], ["I"]),
            (["A", "J"], ["I"]),
        ]
        toy.network = ne._load_tuple_network(rxns)
        toy.convertToIrreversible()
        toy._ensure_rust_ready()
        cls.toy = toy
        cls.ra = toy._rust_arrays

    def test_toy_batch_vs_single(self):
        """Batch of different seed sets on toy network."""
        seed_sets = [
            ["A", "B", "D", "H"],
            ["I"],
            ["A"],
            ["A", "B", "D", "H", "J"],
        ]

        n_compounds = self.ra["n_compounds"]
        x_batch_in = np.zeros((len(seed_sets), n_compounds), dtype=np.uint8)
        for i, seeds in enumerate(seed_sets):
            x_batch_in[i] = self.toy.initialize_metabolite_vector(seeds).astype(np.uint8)

        x_batch_out, y_batch_out = netexprs.expand_batch(
            self.ra["rt_data"], self.ra["rt_indices"], self.ra["rt_indptr"],
            self.ra["n_reactions"],
            self.ra["p_data"], self.ra["p_indices"], self.ra["p_indptr"],
            self.ra["n_compounds"],
            x_batch_in, self.ra["b"],
        )

        for i in range(len(seed_sets)):
            x_single, y_single = netexprs.expand(
                self.ra["rt_data"], self.ra["rt_indices"], self.ra["rt_indptr"],
                self.ra["n_reactions"],
                self.ra["p_data"], self.ra["p_indices"], self.ra["p_indptr"],
                self.ra["n_compounds"],
                x_batch_in[i], self.ra["b"],
            )
            np.testing.assert_array_equal(x_single, x_batch_out[i],
                                          err_msg=f"x mismatch for seed set {i}: {seed_sets[i]}")
            np.testing.assert_array_equal(y_single, y_batch_out[i],
                                          err_msg=f"y mismatch for seed set {i}: {seed_sets[i]}")


# ---------------------------------------------------------------------------
# High-level tests: _run_expansions_rust vs _run_expansions_python
# ---------------------------------------------------------------------------

class TestRunExpansionsBatchHighLevel(unittest.TestCase):
    """Compare _run_expansions_rust (batch) vs _run_expansions_python (loop)."""

    @classmethod
    def setUpClass(cls):
        cls.kegg = ne.GlobalMetabolicNetwork()
        cls.kegg.pruneInconsistentReactions()
        cls.kegg.convertToIrreversible()

        seeds_df = ne.pd.read_csv(ne.asset_path + "/compounds/seeds.Goldford2022.csv")
        cls.all_seeds = list(set(seeds_df["ID"]))

    def test_kegg_batch_parity(self):
        """Multiple seed sets on KEGG should match Python loop."""
        random_seed(42)
        seed_sets = [
            self.all_seeds,
            self.all_seeds[:10],
            sample(self.all_seeds, 20),
            sample(self.all_seeds, 5),
            ["C00001"],
        ]

        cpds_rust, rxns_rust = self.kegg._run_expansions_rust(seed_sets)
        cpds_py, rxns_py = self.kegg._run_expansions_python(seed_sets, algorithm="naive")

        self.assertEqual(len(cpds_rust), len(cpds_py))
        for i in range(len(seed_sets)):
            self.assertEqual(set(cpds_rust[i]), set(cpds_py[i]),
                             f"Compound mismatch for seed set {i}")
            self.assertEqual(set(rxns_rust[i]), set(rxns_py[i]),
                             f"Reaction mismatch for seed set {i}")

    def test_kegg_many_random_seeds(self):
        """Larger batch of random seed subsets."""
        random_seed(123)
        n_batches = 50
        seed_sets = []
        for _ in range(n_batches):
            n = np.random.randint(1, len(self.all_seeds))
            seed_sets.append(sample(self.all_seeds, n))

        cpds_rust, rxns_rust = self.kegg._run_expansions_rust(seed_sets)
        cpds_py, rxns_py = self.kegg._run_expansions_python(seed_sets, algorithm="naive")

        for i in range(n_batches):
            self.assertEqual(set(cpds_rust[i]), set(cpds_py[i]),
                             f"Compound mismatch for seed set {i}")
            self.assertEqual(set(rxns_rust[i]), set(rxns_py[i]),
                             f"Reaction mismatch for seed set {i}")


class TestRunExpansionsBatchHighLevelToy(unittest.TestCase):
    """High-level batch tests on a toy network."""

    @classmethod
    def setUpClass(cls):
        cls.toy = ne.GlobalMetabolicNetwork("dev")
        rxns = [
            (["A", "B"], ["C"]),
            (["C", "D"], ["E", "F"]),
            (["E", "F"], ["G"]),
            (["G", "H"], ["I"]),
            (["A", "J"], ["I"]),
        ]
        cls.toy.network = ne._load_tuple_network(rxns)
        cls.toy.convertToIrreversible()

    def test_toy_batch_parity(self):
        """Toy network batch should match Python loop."""
        seed_sets = [
            ["A", "B", "D", "H"],
            ["I"],
            ["A"],
            ["A", "B", "D", "H", "J"],
        ]

        cpds_rust, rxns_rust = self.toy._run_expansions_rust(seed_sets)
        cpds_py, rxns_py = self.toy._run_expansions_python(seed_sets, algorithm="naive")

        for i in range(len(seed_sets)):
            self.assertEqual(set(cpds_rust[i]), set(cpds_py[i]),
                             f"Compound mismatch for seed set {i}: {seed_sets[i]}")
            self.assertEqual(set(rxns_rust[i]), set(rxns_py[i]),
                             f"Reaction mismatch for seed set {i}: {seed_sets[i]}")


# ---------------------------------------------------------------------------
# Dispatch tests: run_expansions and run_expansions_parallel
# ---------------------------------------------------------------------------

class TestRunExpansionsDispatch(unittest.TestCase):
    """Verify that run_expansions dispatches to _run_expansions_rust."""

    @classmethod
    def setUpClass(cls):
        cls.kegg = ne.GlobalMetabolicNetwork()
        cls.kegg.pruneInconsistentReactions()
        cls.kegg.convertToIrreversible()

        seeds_df = ne.pd.read_csv(ne.asset_path + "/compounds/seeds.Goldford2022.csv")
        cls.all_seeds = list(set(seeds_df["ID"]))

    @unittest.skipIf(netexprs is None, "netexprs not installed")
    def test_run_expansions_uses_batch(self):
        """run_expansions with naive should use _run_expansions_rust."""
        from unittest.mock import patch

        seed_sets = [self.all_seeds[:10], self.all_seeds[:20]]

        with patch.object(self.kegg, '_run_expansions_rust',
                          wraps=self.kegg._run_expansions_rust) as mock_rust, \
             patch.object(self.kegg, '_run_expansions_python',
                          wraps=self.kegg._run_expansions_python) as mock_py:

            with patch.object(ne, '_HAS_RUST', True):
                self.kegg.run_expansions(seed_sets, algorithm="naive")

            mock_rust.assert_called_once()
            mock_py.assert_not_called()

    def test_run_expansions_trace_uses_python(self):
        """run_expansions with trace should use _run_expansions_python."""
        from unittest.mock import patch

        seed_sets = [self.all_seeds[:10]]

        with patch.object(self.kegg, '_run_expansions_rust',
                          wraps=self.kegg._run_expansions_rust) as mock_rust, \
             patch.object(self.kegg, '_run_expansions_python',
                          wraps=self.kegg._run_expansions_python) as mock_py:

            with patch.object(ne, '_HAS_RUST', True):
                self.kegg.run_expansions(seed_sets, algorithm="trace")

            mock_rust.assert_not_called()
            mock_py.assert_called_once()

    def test_run_expansions_parallel_delegates(self):
        """run_expansions_parallel should produce same results as run_expansions."""
        random_seed(42)
        seed_sets = [
            self.all_seeds,
            self.all_seeds[:10],
            sample(self.all_seeds, 20),
        ]

        cpds_normal, rxns_normal = self.kegg.run_expansions(seed_sets)
        cpds_parallel, rxns_parallel = self.kegg.run_expansions_parallel(seed_sets)

        for i in range(len(seed_sets)):
            self.assertEqual(set(cpds_normal[i]), set(cpds_parallel[i]),
                             f"Compound mismatch for seed set {i}")
            self.assertEqual(set(rxns_normal[i]), set(rxns_parallel[i]),
                             f"Reaction mismatch for seed set {i}")


if __name__ == "__main__":
    unittest.main()