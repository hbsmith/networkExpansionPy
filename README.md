# networkExpansionPy

Python package to construct biosphere-level metabolic networks and run network expansion algorithms. This package contains functions to prune biochemical reactions based on thermodynamic constraints and simulate the metabolic scope of different chemical environments.

## Overview

The package is built from algorithms described in the following papers:

**Goldford J.E. et al**, *Remnants of an ancient metabolism without phosphate.* Cell 168, 1–9, March 9, 2017

**Goldford J.E. et al**, *Environmental boundary conditions for the origin of life converge to an organo-sulfur metabolism.* Nature Ecol Evo 3,12 1715-1724, November 11, 2019

## Installation

### Basic Installation

Clone the repository and install using pip in a conda or virtual environment:

```bash
git clone https://github.com/jgoldford/networkExpansionPy.git
cd networkExpansionPy
pip install -e .
```

### Rust Acceleration (Recommended)

For **50-100x speedup** on batch operations, install the optional Rust backend:

```bash
pip install netexprs
```

The package automatically detects and uses the Rust backend when available, with seamless fallback to pure Python if not installed. No code changes required!

#### Performance Comparison

Benchmarks on KEGG metabolic network (8,743 compounds, 18,940 reactions):

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Batch Masked Expansion (n=10) | 21.6 s | 256 ms | **84x** |
| Batch Contraction (n=10) | 3.52 s | 63 ms | **56x** |
| Single Masked Expansion | 2.17 s | 36 ms | **61x** |
| Single Expansion | 281 ms | 222 ms | 1.3x |
| Single Contraction | 25 ms | 15 ms | 1.7x |

The Rust backend provides massive speedups for batch operations through parallel processing, making large-scale robustness analyses and parameter sweeps practical.

## Quick Start

### Basic Network Expansion

```python
import networkExpansionPy.lib as ne

# Load metabolic network
kegg = ne.GlobalMetabolicNetwork()
kegg.pruneInconsistentReactions()
kegg.convertToIrreversible()

# Define seed compounds
seeds = ["C00001", "C00002", "C00003"]  # Example compound IDs

# Run expansion
compound_scope, reaction_scope = kegg.expand(seeds)

print(f"Starting from {len(seeds)} compounds")
print(f"Expanded to {len(compound_scope)} compounds")
print(f"Using {len(reaction_scope)} reactions")
```

### Batch Masked Expansions

Test network robustness by removing different reaction sets:

```python
# Define multiple reaction masks (reactions to remove)
masked_reaction_sets = [
    ["R00001", "R00002"],  # Remove these reactions
    ["R00003", "R00004"],  # Or these
    # ... many more ...
]

# Run all expansions in parallel (automatic with Rust backend)
compound_scopes, reaction_scopes = kegg.run_expansions_reactionMasks_parallel(
    seeds, 
    masked_reaction_sets
)

# Analyze results
for i, (cpds, rxns) in enumerate(zip(compound_scopes, reaction_scopes)):
    print(f"Mask {i}: {len(cpds)} compounds, {len(rxns)} reactions")
```

### Network Contraction

Remove reactions and contract the network:

```python
# Start with a full expansion
compound_scope, reaction_scope = kegg.expand(seeds)

# Define reactions to extinct
extinct_reactions = ["R00001", "R00002"]

# Contract the network
remaining_compounds, remaining_reactions = kegg.contract(
    seeds,
    reaction_scope,
    compound_scope, 
    extinct_reactions
)

print(f"After extinction: {len(remaining_compounds)} compounds remain")
```

### Trace Algorithm

Track compound and reaction addition at each iteration:

```python
compound_dict, reaction_dict = kegg.expand(seeds, algorithm='trace')

# compound_dict and reaction_dict map IDs to iteration numbers
print(f"C00001 appeared at iteration {compound_dict['C00001']}")
print(f"R00001 appeared at iteration {reaction_dict['R00001']}")
```

## API Overview

### Core Methods

#### `expand(seedSet, algorithm='naive')`
Run network expansion from seed compounds.
- **Args**:
  - `seedSet`: List of compound IDs to start from
  - `algorithm`: Either `'naive'` (fast) or `'trace'` (tracks iterations)
- **Returns**: `(compound_scope, reaction_scope)` or `(compound_dict, reaction_dict)` for trace

#### `contract(seedSet, reactionScope, compoundScope, extinctReactions)`
Contract network after removing reactions.
- **Args**:
  - `seedSet`: Initial seed compounds
  - `reactionScope`: Full reaction scope before contraction
  - `compoundScope`: Full compound scope before contraction
  - `extinctReactions`: List of reactions to remove
- **Returns**: `(remaining_compounds, remaining_reactions)`

#### `run_expansions_reactionMasks_parallel(seedSet, maskedReactionSets)`
Run multiple masked expansions in parallel.
- **Args**:
  - `seedSet`: Initial seed compounds
  - `maskedReactionSets`: List of reaction sets to remove (one per mask)
- **Returns**: `(compound_scopes, reaction_scopes)` - lists of results

#### `run_contractions(seedSet, reactionScope, compoundScope, extinctReactionSets)`
Run multiple contractions in parallel.
- **Args**:
  - Similar to contract but accepts list of extinction sets
- **Returns**: `(compound_scopes, reaction_scopes)` - lists of results

## Backend Selection

The package automatically selects the best available backend:

| Condition | Backend Used |
|-----------|--------------|
| `netexprs` installed + naive algorithm | **Rust** (fast) |
| `netexprs` not installed | Python |
| Trace algorithm requested | Python |

You can check which backend is active:

```python
import networkExpansionPy.lib as ne

if ne._HAS_RUST:
    print("Using Rust acceleration")
else:
    print("Using pure Python")
```

### When Rust is NOT Used

Even with `netexprs` installed, Python backend is used for:
- **Trace algorithm**: Requires tracking iteration-by-iteration changes
- **Multi-seed parallel expansions**: Not yet optimized in Rust

All other operations use Rust when available.

## Data Files

The package includes curated datasets in `networkExpansionPy/assets/`:

### Metabolic Networks
- `metabolism.v8.01May2023.pkl`: KEGG-based global metabolic network
- Pre-processed networks with thermodynamic constraints

### Compound Sets
- `seeds.Goldford2022.csv`: Geochemically plausible seed compounds
- `seeds.*.csv`: Various seed compound collections

## Testing

### Unit Tests
```bash
python -m pytest tests/
```

## Benchmarking

Compare Python vs Rust performance:

```bash
python benchmark/benchmark_rust_vs_python.py
```

Results are saved/appended to `benchmark_results.json`.

## Advanced Usage

### Custom Network Loading

```python
import pandas as pd

# Load custom network from DataFrame
# Required columns: 'cid' (compound), 'rn' (reaction), 's' (stoichiometry)
network_df = pd.read_csv('my_network.csv')

kegg = ne.GlobalMetabolicNetwork()
kegg.network = network_df
kegg.pruneInconsistentReactions()
```

### Working with Matrices Directly

```python
# Get stoichiometric matrices
R, P = kegg.create_RP_from_irreversible_network()
# R: reactant matrix (compounds × reactions)
# P: product matrix (compounds × reactions)

# Run low-level expansion
from scipy.sparse import csr_matrix
x0 = kegg.initialize_metabolite_vector(seeds)
b = np.sum(R, axis=0)

x_final, y_final = ne.netExp(csr_matrix(R), csr_matrix(P), 
                               csr_matrix(x0).T, csr_matrix(b).T)
```