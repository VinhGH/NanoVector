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

## ⚡ SIMD Acceleration (Java Vector API)

NanoVector leverages the **Java Vector API** (`jdk.incubator.vector`) to vectorize vector distance calculations directly to hardware SIMD units (AVX2, AVX-512, NEON):

- **Dynamic Hardware Adaptation**: Leverages `FloatVector.SPECIES_PREFERRED` to automatically adapt to the host CPU's optimal lane count (e.g., 256-bit AVX2 / 8 float lanes, 512-bit AVX-512 / 16 float lanes).
- **SIMD Implementations**:
  - `VectorEuclideanDistance`: Uses `FloatVector.sub()` and `FloatVector.fma()` with `VectorOperators.ADD` lane reduction.
  - `VectorCosineDistance`: Direct vectorized dot product on pre-normalized vectors with non-negative clamping ($1.0f - \text{dot} \ge 0.0f$).
  - `VectorDotProductDistance`: FMA inner product negation conforming to distance minimization.
- **Tail-Loop Invariant**: Computes upper loop bounds with `SPECIES.loopBound(length)` and processes non-aligned remainder dimensions via a scalar tail loop, ensuring exact mathematical equivalence across arbitrary dimensions.
- **Zero-Allocation**: Evaluates vectors directly against the contiguous `VectorStorage` buffer without array allocations.

### 📈 Empirical Benchmark Results (Java 25, AVX2 256-bit / 8 lanes)

#### 1. Raw Distance Calculator Throughput (Buffer-to-Query, 100k calls)

| Metric | Dimension | Scalar Throughput | SIMD Throughput | Measured Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **EUCLIDEAN** | 32 | 14.02 MOps/s | 54.42 MOps/s | **3.88x** |
| **EUCLIDEAN** | 64 | 18.60 MOps/s | 51.58 MOps/s | **2.77x** |
| **EUCLIDEAN** | 100 | 12.45 MOps/s | 33.28 MOps/s | **2.67x** |
| **EUCLIDEAN** | 128 | 9.59 MOps/s | 29.62 MOps/s | **3.09x** |
| **EUCLIDEAN** | 384 | 3.31 MOps/s | 16.35 MOps/s | **4.94x** |
| **EUCLIDEAN** | 768 | 1.73 MOps/s | 9.59 MOps/s | **5.55x** |
| **EUCLIDEAN** | 1536 | 0.87 MOps/s | 4.06 MOps/s | **4.68x** |
| **COSINE** | 32 | 11.07 MOps/s | 13.70 MOps/s | **1.24x** |
| **COSINE** | 64 | 11.70 MOps/s | 19.67 MOps/s | **1.68x** |
| **COSINE** | 100 | 9.79 MOps/s | 17.87 MOps/s | **1.82x** |
| **COSINE** | 128 | 8.16 MOps/s | 16.46 MOps/s | **2.02x** |
| **COSINE** | 384 | 3.47 MOps/s | 8.61 MOps/s | **2.48x** |
| **COSINE** | 768 | 1.72 MOps/s | 9.05 MOps/s | **5.26x** |
| **COSINE** | 1536 | 0.87 MOps/s | 4.45 MOps/s | **5.14x** |
| **DOT_PRODUCT** | 32 | 28.79 MOps/s | 46.71 MOps/s | **1.62x** |
| **DOT_PRODUCT** | 64 | 16.97 MOps/s | 42.11 MOps/s | **2.48x** |
| **DOT_PRODUCT** | 128 | 9.57 MOps/s | 16.64 MOps/s | **1.74x** |
| **DOT_PRODUCT** | 384 | 3.46 MOps/s | 15.06 MOps/s | **4.35x** |
| **DOT_PRODUCT** | 768 | 1.78 MOps/s | 8.75 MOps/s | **4.93x** |
| **DOT_PRODUCT** | 1536 | 0.86 MOps/s | 4.11 MOps/s | **4.76x** |

#### 2. FlatIndex Brute-Force Scan ($N=1{,}000, D=128, k=10, 5{,}000 \text{ queries}$)

| Engine | QPS | Mean Latency | p50 Latency | p95 Latency | p99 Latency | Scan Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flat (Scalar)** | 9,264 | 107.94 μs | 103.90 μs | 119.30 μs | 197.00 μs | 1.00x |
| **Flat (SIMD)** | **31,427** | **31.82 μs** | **29.00 μs** | **41.00 μs** | **50.40 μs** | **3.39x** |

#### 3. HNSW Search & Build ($N=1{,}000, D=128, M=16, efConstruction=200, k=10$)

* **Build Time**: Scalar = 920.91 ms | **SIMD = 429.95 ms (2.14x faster build)**

| `efSearch` | Engine | QPS | Mean Latency | p50 Latency | p99 Latency | Search Speedup |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **10** | HNSW (Scalar) | 22,995 | 43.49 μs | 41.50 μs | 87.00 μs | 1.00x |
| **10** | **HNSW (SIMD)** | **53,349** | **18.74 μs** | **18.00 μs** | **30.10 μs** | **2.32x** |
| **50** | HNSW (Scalar) | 8,487 | 117.82 μs | 112.80 μs | 180.30 μs | 1.00x |
| **50** | **HNSW (SIMD)** | **16,659** | **60.03 μs** | **56.70 μs** | **123.60 μs** | **1.96x** |
| **100** | HNSW (Scalar) | 6,218 | 160.83 μs | 154.70 μs | 283.70 μs | 1.00x |
| **100** | **HNSW (SIMD)** | **10,606** | **94.29 μs** | **90.20 μs** | **134.00 μs** | **1.71x** |

---

## 🚀 Getting Started

### Prerequisites
- JDK 21 or higher (compiled with `--release 21`, compatible with JDK 25+).
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

This runs all 146 unit and integration tests (distance metrics, SIMD/scalar equivalence, storage, heaps, flat index, graph invariants, zero-allocation distance evaluations, and empirical recall verification) across Linux and Windows CI.

### Run Performance Benchmarks
To measure raw distance throughput and SIMD speedups across dimensions and search engines:
```powershell
.\mvnw.cmd test -Dtest=SimdBenchmark
```

To run HNSW build and memory allocation microbenchmarks:
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
- [x] **v0.3 (Phase 3)**: Experimental SIMD acceleration via Java Vector API (`jdk.incubator.vector`) (146 unit tests):
  - [x] Incubator module configuration in Maven compiler, surefire, and GitHub Actions CI.
  - [x] Vectorized distance engines (`VectorEuclideanDistance`, `VectorCosineDistance`, `VectorDotProductDistance`).
  - [x] Hardware-adaptive lane sizing (`FloatVector.SPECIES_PREFERRED`) and scalar tail loop.
  - [x] Comprehensive scalar vs SIMD equivalence test suite across dimensions and metrics.
  - [x] Integration into `FlatIndex` and `HnswIndex` (default SIMD for HNSW, exact scalar oracle for Flat).
  - [x] Comprehensive benchmark suite (`SimdBenchmark`) demonstrating up to 5.55x raw distance speedup, 3.39x brute-force scan speedup, and 2.32x HNSW search speedup.
- [ ] **v0.4 (Phase 4)**: Binary persistence (`.nvec` file format).
- [ ] **v0.5 (Phase 5)**: Rigorous benchmark suite (Recall@K vs Latency, Memory footprint per vector).
- [ ] **v0.6 (Phase 6)**: Standalone CLI & Spring Boot REST API.

