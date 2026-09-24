# NanoVector

An experimental, high-performance in-memory vector similarity search engine written in pure Java (Java 21+), designed to explore cache-friendly memory layouts, vector indexing algorithms (Flat baseline, HNSW), and systems performance engineering.

---

## 🎯 Architecture & Modules

NanoVector is organized as a multi-module Maven project:

```text
NanoVector
│
├── nanovector-core          # Vector storage, distance metrics, heap, FlatIndex, and HNSW
├── nanovector-persistence   # Binary serialization format (.nvec) for saving/loading indexes
├── nanovector-benchmark     # Benchmarking suite (Recall@K, Latency P50/P90/P99, Memory profiling)
├── nanovector-cli           # Standalone command-line interface
└── nanovector-server        # Spring Boot REST API for exposing the search engine
```

---

## 💡 Core Design Principles

1. **Contiguous Primitive Memory Layout**:
   - Vectors are stored in a single flat 1D primitive array (`float[] vectors`, where `offset = internalId * dimension`).
   - Eliminates pointer indirection and cache misses compared to array-of-arrays (`float[][]`).
2. **Distance Minimization Contract**:
   - Unified convention: **smaller distance signifies higher similarity**.
   - **Squared Euclidean ($L_2^2$)**: $\sum (a_i - b_i)^2$ (square root omitted to save CPU cycles).
   - **Cosine Distance**: $1.0f - (u \cdot v)$ on pre-normalized unit vectors.
   - **Dot Product**: $-(u \cdot v)$ (negated to conform to minimization).
3. **Allocation-Free Scan Loop**:
   - Distance computations run directly against the internal contiguous buffer.
   - Zero heap allocations during the distance scan.
4. **Deterministic Tie-Breaking**:
   - When distances are identical, ranking defaults to `externalId` ascending, ensuring 100% reproducible search results.
5. **Exact Ground Truth Baseline**:
   - `FlatIndex` ($O(N)$ sequential scan) acts as the exact reference oracle for measuring recall in approximate nearest neighbor (ANN) graphs.

---

## 🧠 HNSW (Hierarchical Navigable Small World) Engine

`HnswIndex` implements the `VectorIndex` contract to provide **empirically sublinear search performance, with approximately logarithmic behavior under suitable conditions** (*Malkov & Yashunin, 2018*).

### Separation of Concerns Architecture

```text
HnswIndex (VectorIndex contract: insert, searchKnn)
    │
    ├── VectorStorage    (Contiguous float[] buffer, externalId ↔ internalId mapping)
    │
    └── HnswGraph        (Graph topology, entryPoint, connect, prune, multi-layer routing)
            │
            ├── HnswNode         (internalId, maxLevel, int[][] neighbors) [Zero VectorStorage coupling]
            ├── LevelGenerator   (Exponential decay level assignment with deterministic seed)
            ├── NeighborSelector (Algorithm 4 heuristic + deterministic fallback)
            └── EpochVisitedSet  (Epoch-based O(1) visited tracking)
```

### Key Technical Implementations

1. **Neighbor Selection**:
   - Implements the HNSW neighbor-selection heuristic inspired by Algorithm 4, with a deterministic fallback to prevent unnecessarily sparse local neighborhoods.
   - Balances distance minimization with angular diversity, pruning redundant long-range edges when closer alternatives exist.
2. **Epoch-Based Visited Tracking (`EpochVisitedSet`)**:
   - Employs an `int[] visitedEpoch` array incremented per query.
   - Provides **zero allocation and zero clearing overhead during normal search operations**.
3. **Zero-Allocation Distance Path (Decoupled Evaluators)**:
   - Pure graph components (`HnswNode`, `HnswGraph`) do not hold vector data or depend on `VectorStorage`.
   - Distances are evaluated via functional interfaces (`DistanceToQuery`, `NodeDistanceEvaluator`) injected by `HnswIndex`, ensuring strict modularity and testability.
   - Vector distances directly evaluate contiguous storage slices via primitive buffer offsets (`distance(buffer, offsetA, buffer, offsetB, length)`), eliminating all `float[]` heap allocations during node routing and pruning (-98% heap allocation reduction during graph construction).
4. **Graph Invariant Verification**:
   - **Degree Constraints**: Strictly bounded to $\le M$ for layers $l > 0$ and $\le M_0 = 2M$ for layer $0$.
   - **Layer 0 Full Connectivity**: 100% of nodes in the index form a single connected component on layer 0 (verified by BFS).
   - **Bidirectional Edge Symmetry**: Verified 100% ($u \in \text{neighbors}(v, l) \iff v \in \text{neighbors}(u, l)$) across all layers with symmetric pruning.
   - **Entry Point Validity**: Verified entry point is anchored at the maximum assigned graph level.
   - **Deterministic Reproducibility**: Fixed seed configuration produces identical graph topologies and KNN rankings.

---

## 📊 Empirical Recall Verification (HNSW vs FlatIndex Oracle)

Recall@10 was experimentally measured on a synthetic benchmark dataset:
- **Dataset**: $N = 1{,}000$ uniform random vectors, $D = 128$ dimensions.
- **Queries**: $Q = 50$ random queries, $k = 10$.
- **Graph Configuration**: $M = 16, M_0 = 32, efConstruction = 200, \text{seed} = 42$.
- **Ground Truth**: Exact exhaustive top-10 from `FlatIndex`.

### Measured Results:

| `efSearch` | Measured Recall@10 | Notes |
| :---: | :---: | :--- |
| **10** | **70.60%** | Fastest search speed |
| **20** | **87.20%** | Balanced speed / recall |
| **50** | **98.80%** | High-precision retrieval |
| **100** | **100.00%** | Perfect match with Ground Truth Oracle |

> [!NOTE]
> All figures above represent actual measured data from automated verification (`HnswRecallTest`), without rounding or synthetic extrapolation.

---

## 🚀 Getting Started

### Prerequisites
- JDK 21 or higher (compiled with `--release 21`).
- Git.

### Build, Test & Verify
NanoVector uses the Maven Wrapper (`mvnw` / `mvnw.cmd`):

On Windows:
```powershell
.\mvnw.cmd clean verify
```

On Linux / macOS:
```bash
./mvnw clean verify
```

This runs all 74 unit tests (distance metrics, storage, heaps, flat index, graph invariants, zero-allocation distance evaluations, and empirical recall verification) across Linux and Windows CI.

### Run Performance Benchmarks
To measure index build throughput, search latency percentiles (p50, p95, p99), and memory allocation:
```powershell
.\mvnw.cmd test -Dtest=HnswBenchmark
```

---

## 🗺️ Roadmap & Evolutionary Milestones

- [x] **v0.1 (Phase 1)**: Multi-module setup, contiguous `VectorStorage`, distance metrics ($L_2^2$, Cosine, Dot Product), primitive `BoundedMaxHeap`, `FlatIndex` Ground Truth Oracle (26 unit tests).
- [x] **v0.2 (Phase 2)**: HNSW Core Engine conforming to `VectorIndex` contract (74 unit tests):
  - [x] Exponential level distribution generator (`LevelGenerator`).
  - [x] $O(1)$ epoch-based visited tracking (`EpochVisitedSet`).
  - [x] Algorithm 4 heuristic with fallback neighbor selector (`NeighborSelector`).
  - [x] Decoupled multi-layer graph topology (`HnswNode`, `HnswGraph`).
  - [x] Multi-layer greedy routing & `searchLayer` traversal (Algorithm 2).
  - [x] End-to-end `HnswIndex` implementation with dynamic `efSearch`.
  - [x] Zero-allocation distance hot path (eliminated intermediate array copying during node distance calculations).
  - [x] Graph invariant verification (Degree $\le M/M_0$, BFS connectivity, 100% bidirectional symmetry, seed determinism).
  - [x] Empirical Recall@10 verification against `FlatIndex` Oracle.
  - [x] Microbenchmark harness (`HnswBenchmark` measuring latency percentiles, throughput, and heap allocation).
  - [x] Cross-platform GitHub Actions CI (Ubuntu + Windows).
- [ ] **v0.3 (Phase 3)**: Experimental SIMD acceleration via Java Vector API (`jdk.incubator.vector`).
- [ ] **v0.4 (Phase 4)**: Binary persistence (`.nvec` file format).
- [ ] **v0.5 (Phase 5)**: Rigorous benchmark suite (Recall@K vs Latency, Memory footprint per vector).
- [ ] **v0.6 (Phase 6)**: Standalone CLI & Spring Boot REST API.
