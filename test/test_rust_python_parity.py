import unittest
import numpy as np
from scipy.sparse import csr_matrix
import networkExpansionPy.lib as ne
import netexprs
import pandas as pd



def rpxb(m, seedSet):
    """Extract R, P, x0, b matrices from a GlobalMetabolicNetwork."""
    m.rid_to_idx, m.idx_to_rid = m.create_reaction_dicts()
    m.cid_to_idx, m.idx_to_cid = m.create_compound_dicts()
    x0 = m.initialize_metabolite_vector(seedSet)
    R, P = m.create_RP_from_irreversible_network()
    b = sum(R)

    R = csr_matrix(R)
    P = csr_matrix(P)
    b = csr_matrix(b).transpose()
    x0 = csr_matrix(x0).transpose()

    return R, P, x0, b


def netExp_rs(R, P, x, b):
    """Rust wrapper matching netExp signature."""
    R_T = R.T.tocsr()
    P = P.tocsr()

    x_out, y_out = netexprs.expand(
        R_T.data.astype(np.float64),
        R_T.indices.astype(np.int32),
        R_T.indptr.astype(np.int32),
        R_T.shape[0],
        P.data.astype(np.float64),
        P.indices.astype(np.int32),
        P.indptr.astype(np.int32),
        P.shape[0],
        np.asarray(x.toarray()).ravel().astype(np.uint8),
        np.asarray(b.toarray()).ravel().astype(np.float64),
    )

    return x_out, y_out


class TestExpandParity(unittest.TestCase):

    def test_toy_network(self):
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

        R, P, x0, b = rpxb(toy, ["A", "B", "D", "H"])

        x_py, y_py = ne.netExp(R, P, x0, b)
        x_rs, y_rs = netExp_rs(R, P, x0, b)

        np.testing.assert_array_equal(x_rs, x_py.toarray().ravel())
        np.testing.assert_array_equal(y_rs, y_py.toarray().ravel())

    def test_toy_network_different_seed(self):
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

        R, P, x0, b = rpxb(toy, ["I"])

        x_py, y_py = ne.netExp(R, P, x0, b)
        x_rs, y_rs = netExp_rs(R, P, x0, b)

        np.testing.assert_array_equal(x_rs, x_py.toarray().ravel())
        np.testing.assert_array_equal(y_rs, y_py.toarray().ravel())

    def test_toy_network_no_expansion(self):
        """Seed set that cannot fire any reactions."""
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

        R, P, x0, b = rpxb(toy, ["A", "D", "E", "H"])

        x_py, y_py = ne.netExp(R, P, x0, b)
        x_rs, y_rs = netExp_rs(R, P, x0, b)

        np.testing.assert_array_equal(x_rs, x_py.toarray().ravel())
        np.testing.assert_array_equal(y_rs, y_py.toarray().ravel())

def netExp_masked_rs(R, P, x, b, mask):
    """Rust wrapper for masked expansion."""
    R_T = R.T.tocsr()
    P = P.tocsr()

    x_out, y_out = netexprs.expand_masked(
        R_T.data.astype(np.float64),
        R_T.indices.astype(np.int32),
        R_T.indptr.astype(np.int32),
        R_T.shape[0],
        P.data.astype(np.float64),
        P.indices.astype(np.int32),
        P.indptr.astype(np.int32),
        P.shape[0],
        np.asarray(x.toarray()).ravel().astype(np.uint8),
        np.asarray(b.toarray()).ravel().astype(np.float64),
        mask.astype(np.uint8),
    )

    return x_out, y_out


def netExp_masked_py(R, P, x, b, mask):
    """Python masked expansion using diagonal matrix approach."""
    reaction_mask = csr_matrix(np.diag(mask))
    Rstar = R * reaction_mask
    Pstar = P * reaction_mask
    return ne.netExp(Rstar, Pstar, x, b)


class TestExpandMaskedParity(unittest.TestCase):

    def test_toy_network_masked(self):
        """Mask out one reaction."""
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

        R, P, x0, b = rpxb(toy, ["A", "B", "D", "H"])

        # Mask out reaction 0 (forward and reverse)
        n_reactions = R.shape[1]
        mask = np.ones(n_reactions, dtype=np.uint8)
        mask[0] = 0
        mask[1] = 0

        x_py, y_py = netExp_masked_py(R, P, x0, b, mask)
        x_rs, y_rs = netExp_masked_rs(R, P, x0, b, mask)

        np.testing.assert_array_equal(x_rs, x_py.toarray().ravel())
        np.testing.assert_array_equal(y_rs, y_py.toarray().ravel())

    def test_toy_network_half_masked(self):
        """Mask out half the reactions."""
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

        R, P, x0, b = rpxb(toy, ["A", "B", "D", "H"])

        n_reactions = R.shape[1]
        mask = np.ones(n_reactions, dtype=np.uint8)
        mask[: n_reactions // 2] = 0

        x_py, y_py = netExp_masked_py(R, P, x0, b, mask)
        x_rs, y_rs = netExp_masked_rs(R, P, x0, b, mask)

        np.testing.assert_array_equal(x_rs, x_py.toarray().ravel())
        np.testing.assert_array_equal(y_rs, y_py.toarray().ravel())


def load_metabolism(fname):
    return pd.read_pickle(ne.asset_path  + "/metabolic_networks/" + fname)

def load_compounds(fname):
    return pd.read_csv(ne.asset_path  + "/compounds/" + fname)

class TestExpandLargeNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load KEGG network once for all tests in this class."""
        cls.kegg = load_metabolism("metabolism.v8.01May2023.pkl")

    def test_kegg_network(self):
        seedSet = list(set(load_compounds('seeds.Goldford2022.csv')["ID"]))

        R, P, x0, b = rpxb(self.kegg, seedSet)

        x_py, y_py = ne.netExp(R, P, x0, b)
        x_rs, y_rs = netExp_rs(R, P, x0, b)

        np.testing.assert_array_equal(x_rs, x_py.toarray().ravel())
        np.testing.assert_array_equal(y_rs, y_py.toarray().ravel())

    def test_kegg_network_minimal_seed(self):
        """Single compound seed."""
        seedSet = ["C00001"]

        R, P, x0, b = rpxb(self.kegg, seedSet)

        x_py, y_py = ne.netExp(R, P, x0, b)
        x_rs, y_rs = netExp_rs(R, P, x0, b)

        np.testing.assert_array_equal(x_rs, x_py.toarray().ravel())
        np.testing.assert_array_equal(y_rs, y_py.toarray().ravel())


class TestExpandMaskedLargeNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load KEGG network once for all tests in this class."""
        cls.kegg = load_metabolism("metabolism.v8.01May2023.pkl")

    def test_kegg_masked_random(self):
        """Randomly mask 10% of reactions."""
        seedSet = list(set(load_compounds('seeds.Goldford2022.csv')["ID"]))

        R, P, x0, b = rpxb(self.kegg, seedSet)

        n_reactions = R.shape[1]
        np.random.seed(42)
        mask = (np.random.random(n_reactions) > 0.1).astype(np.uint8)

        x_py, y_py = netExp_masked_py(R, P, x0, b, mask)
        x_rs, y_rs = netExp_masked_rs(R, P, x0, b, mask)

        np.testing.assert_array_equal(x_rs, x_py.toarray().ravel())
        np.testing.assert_array_equal(y_rs, y_py.toarray().ravel())

    def test_kegg_masked_half(self):
        """Mask out first half of reactions."""
        seedSet = list(set(load_compounds('seeds.Goldford2022.csv')["ID"]))

        R, P, x0, b = rpxb(self.kegg, seedSet)

        n_reactions = R.shape[1]
        mask = np.ones(n_reactions, dtype=np.uint8)
        mask[: n_reactions // 2] = 0

        x_py, y_py = netExp_masked_py(R, P, x0, b, mask)
        x_rs, y_rs = netExp_masked_rs(R, P, x0, b, mask)

        np.testing.assert_array_equal(x_rs, x_py.toarray().ravel())
        np.testing.assert_array_equal(y_rs, y_py.toarray().ravel())


if __name__ == "__main__":
    unittest.main()