import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)

import numpy as np
from scipy.sparse import csr_matrix
import networkExpansionPy.lib as ne
import netexprs
import pandas as pd
import time
import json
import os
from datetime import datetime
import multiprocessing

# Get the directory containing the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

RESULTS_FILE = os.path.join(current_dir, "benchmark_results.json")

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []

def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

def benchmark(fn, args, n_runs=10, warmup=2):
    for _ in range(warmup):
        fn(*args)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "n_runs": n_runs,
    }

def run_benchmarks():
    # Load data
    print("Loading data...")
    kegg = ne.load_metabolism("metabolism.v8.01May2023.pkl")
    seedSet = list(set(ne.load_compounds('seeds.Goldford2022.csv')["ID"]))
    
    kegg.rid_to_idx, kegg.idx_to_rid = kegg.create_reaction_dicts()
    kegg.cid_to_idx, kegg.idx_to_cid = kegg.create_compound_dicts()
    R, P = kegg.create_RP_from_irreversible_network()
    b = sum(R)
    R = csr_matrix(R)
    P = csr_matrix(P)
    b_sparse = csr_matrix(b).transpose()
    x0 = kegg.initialize_metabolite_vector(seedSet)
    x0_sparse = csr_matrix(x0).transpose()
    
    # Pre-convert for Rust
    R_T = R.T.tocsr()
    rt_data = R_T.data.astype(np.float64)
    rt_indices = R_T.indices.astype(np.int32)
    rt_indptr = R_T.indptr.astype(np.int32)
    n_reactions = R_T.shape[0]
    
    P_csr = P.tocsr()
    p_data = P_csr.data.astype(np.float64)
    p_indices = P_csr.indices.astype(np.int32)
    p_indptr = P_csr.indptr.astype(np.int32)
    n_compounds = P_csr.shape[0]
    
    x0_arr = np.asarray(x0).ravel().astype(np.uint8)
    b_arr = np.asarray(b).ravel().astype(np.float64)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_compounds": n_compounds,
        "n_reactions": n_reactions,
        "n_cores": multiprocessing.cpu_count(),
        "benchmarks": {}
    }
    
    # --- Benchmark 1: Single Expansion ---
    print("\nBenchmarking single expansion...")
    results["benchmarks"]["expand_python"] = benchmark(
        lambda: ne.netExp(R, P, x0_sparse, b_sparse), [], n_runs=10
    )
    results["benchmarks"]["expand_rust"] = benchmark(
        lambda: netexprs.expand(
            rt_data, rt_indices, rt_indptr, n_reactions,
            p_data, p_indices, p_indptr, n_compounds,
            x0_arr.copy(), b_arr
        ), [], n_runs=10
    )

    # --- Benchmark 2: Masked Expansion (Single) ---
    print("Benchmarking single masked expansion...")
    np.random.seed(42)
    mask = (np.random.random(n_reactions) > 0.1).astype(np.uint8)
    
    results["benchmarks"]["expand_masked_python"] = benchmark(
        lambda: ne.netExp(
            R * csr_matrix(np.diag(mask)), 
            P * csr_matrix(np.diag(mask)), 
            x0_sparse, b_sparse
        ),
        [],
        n_runs=10
    )
    
    results["benchmarks"]["expand_masked_rust"] = benchmark(
        lambda: netexprs.expand_masked(
            rt_data, rt_indices, rt_indptr, n_reactions,
            p_data, p_indices, p_indptr, n_compounds,
            x0_arr.copy(), b_arr, mask
        ),
        [],
        n_runs=10
    )

    # --- Benchmark 3: Batch Masked Expansion ---
    for n_masks in [10]:
        print(f"Benchmarking batch masked expansion (n={n_masks})...")
        
        np.random.seed(123)
        masks = (np.random.random((n_masks, n_reactions)) > 0.1).astype(np.uint8)
        
        # Convert masks to the format run_expansions_reactionMasks_parallel expects
        masked_reaction_sets = []
        for i in range(n_masks):
            removed_rxns = [kegg.idx_to_rid[j] for j in range(n_reactions) if masks[i, j] == 0]
            masked_reaction_sets.append(removed_rxns)
        
        # Python parallel
        results["benchmarks"][f"expand_batch_{n_masks}_python_parallel"] = benchmark(
            lambda: kegg._run_expansions_reactionMasks_parallel_python(seedSet, masked_reaction_sets),
            [],
            n_runs=3
        )
        
        # Rust batch parallel
        results["benchmarks"][f"expand_batch_{n_masks}_rust_parallel"] = benchmark(
            lambda: netexprs.expand_masked_batch(
                rt_data, rt_indices, rt_indptr, n_reactions,
                p_data, p_indices, p_indptr, n_compounds,
                x0_arr.copy(), b_arr, masks
            ),
            [],
            n_runs=3
        )

    # --- Setup for Contraction ---
    print("\nRunning initial expansion to setup contraction benchmarks...")
    # Get a valid full scope to contract from
    x_expanded_sparse, y_expanded_sparse = ne.netExp(R, P, x0_sparse, b_sparse)
    
    # Convert to arrays for Rust
    x_expanded_arr = np.asarray(x_expanded_sparse.toarray()).ravel().astype(np.uint8)
    y_expanded_arr = np.asarray(y_expanded_sparse.toarray()).ravel().astype(np.uint8)
    
    # Convert to lists of IDs for Python high-level functions
    cidx = np.nonzero(x_expanded_sparse.toarray().T[0])[0]
    compoundScope = [kegg.idx_to_cid[i] for i in cidx]
    
    ridx = np.nonzero(y_expanded_sparse.toarray().T[0])[0]
    reactionScope = [kegg.idx_to_rid[i] for i in ridx]
    
    # --- Benchmark 4: Single Contraction ---
    print("Benchmarking single contraction...")
    
    # Generate random extinction (5% of reactions)
    np.random.seed(55)
    y_extinct_arr = (np.random.random(n_reactions) < 0.05).astype(np.uint8)
    y_extinct_sparse = csr_matrix(y_extinct_arr).T
    
    results["benchmarks"]["contract_python"] = benchmark(
        lambda: ne.netContract(R, P, b_sparse, x_expanded_sparse, y_expanded_sparse, y_extinct_sparse),
        [], n_runs=10
    )
    
    results["benchmarks"]["contract_rust"] = benchmark(
        lambda: netexprs.contract(
            rt_data, rt_indices, rt_indptr, n_reactions,
            p_data, p_indices, p_indptr, n_compounds,
            x_expanded_arr, y_expanded_arr, y_extinct_arr
        ),
        [], n_runs=10
    )
    
    # --- Benchmark 5: Batch Contraction ---
    for n_batches in [10]:
        print(f"Benchmarking batch contraction (n={n_batches})...")
        
        np.random.seed(999)
        # Create N random extinction sets
        y_extinct_batch = (np.random.random((n_batches, n_reactions)) < 0.05).astype(np.uint8)
        
        # Prepare list of lists of IDs for Python
        extinct_reaction_sets = []
        for i in range(n_batches):
            # Extract IDs of extinct reactions
            extinct_indices = np.where(y_extinct_batch[i] == 1)[0]
            extinct_ids = [kegg.idx_to_rid[idx] for idx in extinct_indices]
            extinct_reaction_sets.append(extinct_ids)
            
        # Benchmark Python (using high-level wrapper)
        results["benchmarks"][f"contract_batch_{n_batches}_python"] = benchmark(
            lambda: kegg._run_contractions_python(reactionScope, compoundScope, extinct_reaction_sets),
            [], n_runs=3
        )
        
        # Benchmark Rust (using batch function)
        results["benchmarks"][f"contract_batch_{n_batches}_rust"] = benchmark(
            lambda: netexprs.contract_batch(
                rt_data, rt_indices, rt_indptr, n_reactions,
                p_data, p_indices, p_indptr, n_compounds,
                x_expanded_arr, y_expanded_arr, y_extinct_batch
            ),
            [], n_runs=3
        )

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Cores: {results['n_cores']}")
    print(f"Compounds: {results['n_compounds']}, Reactions: {results['n_reactions']}")
    print()
    
    for name, stats in results["benchmarks"].items():
        print(f"{name:<35}: {stats['mean']*1000:.2f}ms (±{stats['std']*1000:.2f}ms)")
        
    print("\n" + "-"*60)
    print("SPEEDUPS (Python / Rust)")
    print("-"*60)
    
    pairs = [
        ("expand_python", "expand_rust", "Single Expansion"),
        ("expand_masked_python", "expand_masked_rust", "Masked Expansion"),
        ("expand_batch_10_python_parallel", "expand_batch_10_rust_parallel", "Batch Expansion (n=10)"),
        ("contract_python", "contract_rust", "Single Contraction"),
        ("contract_batch_10_python", "contract_batch_10_rust", "Batch Contraction (n=10)")
    ]
    
    for py, rs, label in pairs:
        if py in results["benchmarks"] and rs in results["benchmarks"]:
            speedup = results["benchmarks"][py]["mean"] / results["benchmarks"][rs]["mean"]
            print(f"{label:<30}: {speedup:.2f}x")

    # Save results
    all_results = load_results()
    all_results.append(results)
    save_results(all_results)
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_benchmarks()