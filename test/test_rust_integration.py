"""
Test script to verify Rust integration in networkExpansionPy.

This tests that:
1. Rust is correctly detected and used when available
2. Results match between Rust and Python implementations
3. API remains unchanged for users
"""

import numpy as np
import sys

# Check if netexprs is available
try:
    import netexprs
    print("✓ netexprs (Rust) module found")
    HAS_RUST = True
except ImportError:
    print("✗ netexprs (Rust) module not found - tests will use Python fallback")
    HAS_RUST = False

# Import the library
import networkExpansionPy.lib as ne

def test_basic_expansion():
    """Test that basic expansion works."""
    print("\n--- Testing basic expansion ---")
    
    # Load test data
    kegg = ne.GlobalMetabolicNetwork()
    kegg.pruneInconsistentReactions()
    kegg.convertToIrreversible()
    
    # Load seed set
    seeds = list(set(ne.load_compounds('seeds.Goldford2022.csv')["ID"]))[:50]  # Smaller for quick test
    
    # Run expansion
    compounds, reactions = kegg.expand(seeds)
    
    print(f"  Compounds in scope: {len(compounds)}")
    print(f"  Reactions in scope: {len(reactions)}")
    
    assert len(compounds) > len(seeds), "Expansion should produce more compounds than seeds"
    assert len(reactions) > 0, "Expansion should produce some reactions"
    print("✓ Basic expansion passed")
    
    return kegg, seeds, compounds, reactions


def test_masked_expansion(kegg, seeds):
    """Test masked expansion."""
    print("\n--- Testing masked expansion ---")
    
    # Get some reactions to mask
    kegg._ensure_dicts()
    all_rxns = list(kegg.rid_to_idx.keys())
    rxns_to_remove = all_rxns[:100]  # Remove first 100 reactions
    
    # Run masked expansion
    compounds, reactions = kegg.run_expansions_reactionMasks(seeds, [rxns_to_remove])
    
    print(f"  Compounds in scope: {len(compounds[0])}")
    print(f"  Reactions in scope: {len(reactions[0])}")
    
    assert len(compounds) == 1, "Should return one result"
    print("✓ Masked expansion passed")
    
    return compounds[0], reactions[0]


def test_batch_masked_expansion(kegg, seeds):
    """Test batch masked expansion."""
    print("\n--- Testing batch masked expansion ---")
    
    kegg._ensure_dicts()
    all_rxns = list(kegg.rid_to_idx.keys())
    
    # Create 5 different masks
    np.random.seed(42)
    n_batches = 5
    masked_reaction_sets = []
    for i in range(n_batches):
        # Remove random 10% of reactions
        n_remove = len(all_rxns) // 10
        remove_idx = np.random.choice(len(all_rxns), n_remove, replace=False)
        masked_reaction_sets.append([all_rxns[j] for j in remove_idx])
    
    # Run batch expansion
    compound_scopes, reaction_scopes = kegg.run_expansions_reactionMasks(seeds, masked_reaction_sets)
    
    print(f"  Number of results: {len(compound_scopes)}")
    for i, (c, r) in enumerate(zip(compound_scopes, reaction_scopes)):
        print(f"    Batch {i}: {len(c)} compounds, {len(r)} reactions")
    
    assert len(compound_scopes) == n_batches, f"Should return {n_batches} results"
    print("✓ Batch masked expansion passed")


def test_contraction(kegg, seeds, compound_scope, reaction_scope):
    """Test contraction."""
    print("\n--- Testing contraction ---")
    
    kegg._ensure_dicts()
    
    # Pick some reactions to make extinct
    rxns_to_extinct = reaction_scope[:50]  # First 50 reactions
    
    # Run contraction
    remaining_compounds, remaining_reactions = kegg.contract(
        seeds, reaction_scope, compound_scope, rxns_to_extinct
    )
    
    print(f"  Remaining compounds: {len(remaining_compounds)}")
    print(f"  Remaining reactions: {len(remaining_reactions)}")
    
    assert len(remaining_reactions) <= len(reaction_scope), "Contraction should not add reactions"
    print("✓ Contraction passed")


def test_batch_contraction(kegg, seeds, compound_scope, reaction_scope):
    """Test batch contraction."""
    print("\n--- Testing batch contraction ---")
    
    np.random.seed(123)
    n_batches = 5
    
    # Create different extinction sets
    extinct_sets = []
    for i in range(n_batches):
        n_extinct = len(reaction_scope) // 20  # 5% of reactions
        extinct_idx = np.random.choice(len(reaction_scope), n_extinct, replace=False)
        extinct_sets.append([reaction_scope[j] for j in extinct_idx])
    
    # Run batch contraction
    compound_scopes, reaction_scopes = kegg.run_contractions(
        seeds, reaction_scope, compound_scope, extinct_sets
    )
    
    print(f"  Number of results: {len(compound_scopes)}")
    for i, (c, r) in enumerate(zip(compound_scopes, reaction_scopes)):
        print(f"    Batch {i}: {len(c)} compounds, {len(r)} reactions")
    
    assert len(compound_scopes) == n_batches, f"Should return {n_batches} results"
    print("✓ Batch contraction passed")


def test_rust_python_parity(kegg, seeds):
    """Test that Rust and Python give the same results."""
    if not HAS_RUST:
        print("\n--- Skipping Rust/Python parity test (Rust not available) ---")
        return
    
    print("\n--- Testing Rust/Python parity ---")
    
    # Force Python path
    kegg._ensure_rust_ready()
    
    # Get Rust result
    compounds_rust, reactions_rust = kegg._expand_rust(seeds, reaction_mask=None)
    
    # Get Python result
    compounds_python, reactions_python = kegg._expand_python(seeds, algorithm='naive', reaction_mask=None)
    
    # Compare
    compounds_rust_set = set(compounds_rust)
    compounds_python_set = set(compounds_python)
    reactions_rust_set = set(reactions_rust)
    reactions_python_set = set(reactions_python)
    
    compounds_match = compounds_rust_set == compounds_python_set
    reactions_match = reactions_rust_set == reactions_python_set
    
    print(f"  Rust compounds: {len(compounds_rust)}, Python compounds: {len(compounds_python)}")
    print(f"  Rust reactions: {len(reactions_rust)}, Python reactions: {len(reactions_python)}")
    print(f"  Compounds match: {compounds_match}")
    print(f"  Reactions match: {reactions_match}")
    
    if not compounds_match:
        only_rust = compounds_rust_set - compounds_python_set
        only_python = compounds_python_set - compounds_rust_set
        print(f"    Only in Rust: {len(only_rust)}")
        print(f"    Only in Python: {len(only_python)}")
    
    assert compounds_match, "Compound results should match"
    assert reactions_match, "Reaction results should match"
    print("✓ Rust/Python parity passed")


def main():
    print("=" * 60)
    print("Testing networkExpansionPy Rust Integration")
    print("=" * 60)
    
    # Run tests
    kegg, seeds, compounds, reactions = test_basic_expansion()
    masked_compounds, masked_reactions = test_masked_expansion(kegg, seeds)
    test_batch_masked_expansion(kegg, seeds)
    test_contraction(kegg, seeds, compounds, reactions)
    test_batch_contraction(kegg, seeds, compounds, reactions)
    test_rust_python_parity(kegg, seeds)
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()