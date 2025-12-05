# benchmark_rust_vs_python.py

import time
import json
import os
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix
import networkExpansionPy.lib as ne
import netexprs
import multiprocessing
import pandas as pd

RESULTS_FILE = "benchmark_results.json"

def load_metabolism(fname):
    return pd.read_pickle(ne.asset_path  + "/metabolic_networks/" + fname)

def load_compounds(fname):
    return pd.read_csv(ne.asset_path  + "/compounds/" + fname)


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def benchmark(fn, args, n_runs=10, warmup=2):
    """Run function multiple times and return timing stats."""
    # Warmup
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
    # Load data once
    kegg = load_metabolism("metabolism.v8.01May2023.pkl")
    seedSet = list(set(load_compounds('seeds.Goldford2022.csv')["ID"]))
    
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
    
    # Benchmark 1: Single expansion
    print("Benchmarking single expansion...")
    
    results["benchmarks"]["expand_python"] = benchmark(
        lambda: ne.netExp(R, P, x0_sparse, b_sparse),
        [],
        n_runs=20
    )
    
    results["benchmarks"]["expand_rust"] = benchmark(
        lambda: netexprs.expand(
            rt_data, rt_indices, rt_indptr, n_reactions,
            p_data, p_indices, p_indptr, n_compounds,
            x0_arr.copy(), b_arr
        ),
        [],
        n_runs=20
    )
    
    # Benchmark 2: Masked expansion (single)
    print("Benchmarking single masked expansion...")
    
    np.random.seed(42)
    mask = (np.random.random(n_reactions) > 0.1).astype(np.uint8)
    reaction_mask_matrix = csr_matrix(np.diag(mask))
    Rstar = R * reaction_mask_matrix
    Pstar = P * reaction_mask_matrix
    
    results["benchmarks"]["expand_masked_python"] = benchmark(
        lambda: ne.netExp(Rstar, Pstar, x0_sparse, b_sparse),
        [],
        n_runs=20
    )
    
    results["benchmarks"]["expand_masked_rust"] = benchmark(
        lambda: netexprs.expand_masked(
            rt_data, rt_indices, rt_indptr, n_reactions,
            p_data, p_indices, p_indptr, n_compounds,
            x0_arr.copy(), b_arr, mask
        ),
        [],
        n_runs=20
    )
    
    # Benchmark 3: Batch masked expansion
    for n_masks in [10, 100, 1000]:
        print(f"Benchmarking batch masked expansion (n={n_masks})...")
        
        np.random.seed(123)
        masks = (np.random.random((n_masks, n_reactions)) > 0.1).astype(np.uint8)
        
        # Convert masks to the format run_expansions_reactionMasks_parallel expects
        # It expects lists of reaction tuples to remove, not binary masks
        masked_reaction_sets = []
        for i in range(n_masks):
            removed_rxns = [kegg.idx_to_rid[j] for j in range(n_reactions) if masks[i, j] == 0]
            masked_reaction_sets.append(removed_rxns)
        
        # Python parallel
        results["benchmarks"][f"expand_batch_{n_masks}_python_parallel"] = benchmark(
            lambda: kegg.run_expansions_reactionMasks_parallel(seedSet, masked_reaction_sets),
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
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Cores: {results['n_cores']}")
    print(f"Compounds: {results['n_compounds']}, Reactions: {results['n_reactions']}")
    print()
    
    for name, stats in results["benchmarks"].items():
        print(f"{name}: {stats['mean']*1000:.2f}ms (±{stats['std']*1000:.2f}ms)")
    
    # Calculate speedups
    print("\n" + "-"*60)
    print("SPEEDUPS (Python / Rust)")
    print("-"*60)
    
    if "expand_python" in results["benchmarks"] and "expand_rust" in results["benchmarks"]:
        speedup = results["benchmarks"]["expand_python"]["mean"] / results["benchmarks"]["expand_rust"]["mean"]
        print(f"Single expansion: {speedup:.2f}x")
    
    if "expand_masked_python" in results["benchmarks"] and "expand_masked_rust" in results["benchmarks"]:
        speedup = results["benchmarks"]["expand_masked_python"]["mean"] / results["benchmarks"]["expand_masked_rust"]["mean"]
        print(f"Single masked expansion: {speedup:.2f}x")
    
    for n_masks in [10, 100, 1000]:
        py_key = f"expand_batch_{n_masks}_python_parallel"
        rs_key = f"expand_batch_{n_masks}_rust_parallel"
        if py_key in results["benchmarks"] and rs_key in results["benchmarks"]:
            speedup = results["benchmarks"][py_key]["mean"] / results["benchmarks"][rs_key]["mean"]
            print(f"Batch {n_masks} masks: {speedup:.2f}x")
    
    # Save results
    all_results = load_results()
    all_results.append(results)
    save_results(all_results)
    print(f"\nResults saved to {RESULTS_FILE}")
    
    return results


if __name__ == "__main__":
    run_benchmarks()