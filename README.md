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
├── nanovector-benchmark     # Benchmarking suite (Recall@K, Latency P50/P95/P99, Memory profiling)
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
3. **Allocation-Free Distance Scan**:
   - Distance computations run directly against the internal contiguous buffer.
   - Zero heap allocations during the distance scan.
4. **Deterministic Tie-Breaking**:
   - When distances are identical, ranking defaults to `externalId` ascending, ensuring 100% reproducible search results.
5. **Exact Reference Baseline**:
   - `FlatIndex` ($O(N)$ sequential scan) acts as the exact reference oracle for measuring recall in approximate nearest neighbor (ANN) graphs under NanoVector's floating-point metric implementation.

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
3. **Zero-Allocation Distance Evaluation Path (Decoupled Evaluators)**:
   - Pure graph components (`HnswNode`, `HnswGraph`) do not hold vector data or depend on `VectorStorage`.
   - Distances are evaluated via functional interfaces (`DistanceToQuery`, `NodeDistanceEvaluator`) injected by `HnswIndex`, ensuring strict modularity and testability.
   - Vector distances directly evaluate contiguous storage slices via primitive buffer offsets (`distance(buffer, offsetA, buffer, offsetB, length)`), eliminating all `float[]` heap allocations during node routing and pruning.
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
- **Ground Truth**: Exact exhaustive top-10 from `FlatIndex` reference oracle.

### Measured Results (Scalar HNSW vs SIMD HNSW):

| `efSearch` | Scalar Recall@10 | SIMD Recall@10 | Retrieval Quality Preservation |
| :---: | :---: | :---: | :--- |
| **10** | **70.60%** | **70.60%** | Exact match |
| **20** | **87.20%** | **87.20%** | Exact match |
| **50** | **98.80%** | **98.80%** | Exact match |
| **100** | **100.00%** | **100.00%** | Perfect match with Ground Truth Oracle |

> [!NOTE]
> All figures above represent actual measured data from automated verification (`HnswRecallTest`), confirming that SIMD acceleration fully preserved retrieval quality under the tested configuration.

---

## ⚡ SIMD Acceleration (Java Vector API)

NanoVector leverages the **Java Vector API** (`jdk.incubator.vector`) to vectorize vector distance calculations directly to hardware SIMD units (AVX2, AVX-512, NEON):

- **Hardware Adaptation**: Leverages `FloatVector.SPECIES_PREFERRED` to allow the JVM runtime to select the preferred vector species for the host platform (in our benchmark environment: 256-bit AVX2 with 8 float lanes on Java 25).
- **SIMD Implementations**:
  - `VectorEuclideanDistance`: Uses `FloatVector.sub()` and `FloatVector.fma()` with `VectorOperators.ADD` lane reduction. FMA operations give the JVM/JIT an opportunity to lower the multiply-add operation to hardware fused multiply-add instructions on supported CPUs.
  - `VectorCosineDistance`: Direct vectorized dot product on pre-normalized vectors with non-negative clamping ($\max(0.0f, 1.0f - \text{dot})$).
  - `VectorDotProductDistance`: FMA inner product negation conforming to distance minimization.
- **Tail-Loop Invariant**: Computes upper loop bounds with `SPECIES.loopBound(length)` and processes non-aligned remainder dimensions via a scalar tail loop, ensuring numerical equivalence within tolerance across arbitrary dimensions.
- **Zero-Allocation Distance Evaluation**: Evaluates vectors directly against the contiguous `VectorStorage` buffer without heap array allocations in the hot distance calculation path.

> [!TIP]
> **Bottleneck Propagation Insight**: NanoVector's SIMD distance kernels achieved up to 5.55× higher measured throughput than the scalar baseline on the benchmark environment. End-to-end acceleration was lower—3.39× for FlatIndex and 1.71–2.32× for HNSW—demonstrating that graph traversal, visited set lookups, and candidate queue management become increasingly significant portions of total search cost.

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

## 💾 Binary Persistence (NVEC v1 Format)

NanoVector features **NVEC v1 binary persistence with defensive validation and round-trip behavioral verification** (`nanovector-persistence`). It provides high-throughput binary serialization and deserialization for both `FlatIndex` and `HnswIndex` while preserving strict modularity and avoiding unnecessary defensive copies at the storage boundary.

### Architectural Separation & Controlled Bridges

```text
┌────────────────────────────────────────────────────────┐
│                   nanovector-core                      │
│                                                        │
│  ┌──────────────────────┐    ┌──────────────────────┐  │
│  │   VectorDataView     │    │    IndexRestorer     │  │
│  │ (Read-only contract) │    │  (Controlled Bridge) │  │
│  └──────────▲───────────┘    └──────────┬───────────┘  │
│             │                           │              │
│       VectorStorage                 FlatIndex /        │
│    (internal buffers)                HnswIndex         │
└─────────────┼───────────────────────────▲──────────────┘
              │                           │
              │       nanovector-persistence
              │                           │
        ┌─────┴──────────┐         ┌──────┴─────────┐
        │   NvecWriter   │         │   NvecReader   │
        │ (Save to disk) │         │ (Load to core) │
        └────────────────┘         └────────────────┘
```

1. **`VectorDataView` (Read-Only Contract)**:
   - Exposes direct sequential access to `vectorBuffer()` (`float[]`) and `externalIdBuffer()` (`long[]`).
   - The returned buffer is the internal storage buffer and must be treated as **read-only by contract** by callers, eliminating heap cloning during serialization.
2. **`IndexRestorer` (Controlled Internal Bridge)**:
   - Resides in `nanovector-core` (`com.nanovector.core.index`) to allow `NvecReader` to construct indexes directly from restored buffers without exposing public mutating setters or compromising core encapsulation.
3. **Direct Verbatim $O(E)$ HNSW Restoration**:
   - `NvecReader` restores nodes, layer assignments, and neighbor adjacency arrays directly from the serialized topology.
   - **Strictly avoids rebuilding the graph**: No re-running neighbor discovery, heuristic pruning, or `connect()` passes. The graph topology is reconstructed verbatim in $O(E)$ time.

---

### Binary Format Layout (`.nvec`)

The NVEC v1 specification (`FORMAT_SPEC_V1.md`) uses an explicit **Little-Endian** byte ordering as a format choice to guarantee deterministic cross-platform reproducibility across architectures.

| Section | Size | Description |
| :--- | :---: | :--- |
| **Header** | 32 bytes | Magic (`NVEC`), Version (`1`), Endianness (`1`), IndexType (`1`=FLAT, `2`=HNSW), Metric (`1`=L2, `2`=Cosine, `3`=Dot), Dimension ($D$), VectorCount ($N$), Reserved (4B) |
| **Metadata Block** | Variable | 4-byte `metadata_length` prefix.<br>• **FLAT**: `metadata_length = 0` (4 bytes total).<br>• **HNSW**: `metadata_length = 24` prefix + 24-byte payload ($M, M_0, efConstruction, defaultEfSearch, maxLevel, entryPointId$) = 28 bytes total. |
| **Vector Data** | $N \times D \times 4$ bytes | Contiguous IEEE-754 32-bit single-precision floats. |
| **External IDs** | $N \times 8$ bytes | 64-bit signed integers mapping internal slots to external IDs. |
| **HNSW Topology** | Variable | *(HNSW only)* Per-node layer count, and per-layer neighbor count followed by neighbor internal IDs. |
| **CRC32C Footer** | 4 bytes | Hardware-accelerated Castagnoli CRC-32C calculated over bytes $[0, \text{fileSize} - 4)$. |

---

### Bounded-Memory Chunked I/O & Atomic Move

- **Bounded-Memory Chunked I/O**: `NvecWriter` serializes float buffers and external IDs in fixed 8 KB chunks, maintaining a running CRC32C digest without buffering whole files in heap memory.
- **Atomic Replacement with Fallback**: Saves first to a temporary file (`.tmp`) and swaps atomically via `Files.move(..., ATOMIC_MOVE)`. If atomic move is unsupported across filesystem boundaries, falls back safely to `REPLACE_EXISTING`.
- **Pre-Write Invariant Checks**: Rejects non-finite floats (`NaN`, `Infinity`), rejects non-unit vectors under `COSINE` distance without silently mutating data, and enforces topology degree invariants ($\le M, \le M_0$).

---

### 8-Step Defense-in-Depth Validation Pipeline

`NvecReader` enforces strict validation before constructing indexes:

```text
1. File Size Gate      (fileSize >= 40 bytes)
       │
2. Header Validation   (Magic 'NVEC', Version 1, Little-Endian, valid IndexType & Metric)
       │
3. Structural Bounds   (Detects arithmetic overflow and rejects huge dimension/vectorCount before allocation)
       │
4. CRC32C Integrity   (Streams file bytes [0, fileSize-4) and verifies against footer)
       │
5. Metadata Block      (Reads metadata_length; forward-compatible skip for unknown extensions)
       │
6. Direct Restoration  (Reads vector & ID buffers directly into VectorStorage without extra defensive copying)
       │
7. Topology Invariants (HNSW only: validates entryPoint and neighbor ID references < vectorCount)
       │
8. Controlled Rebuild  (Restores index via IndexRestorer bridge)
```

> [!IMPORTANT]
> **Pre-Allocation Structural Bounds**: Checking $(long) vectorCount \times dimension$ against the actual file size before allocating memory prevents out-of-memory (OOM) denial-of-service vulnerabilities caused by malformed headers with valid CRCs.

---

### 📈 Measured Persistence Benchmarks (NVEC v1)

Measured on Windows 11, amd64, Java 25 (targeting `--release 21`), synthetic benchmark dataset ($D=128$, `EUCLIDEAN`):

| Index Type | Vector Count ($N$) | File Size | Bytes / Vector | Save Time | Write Throughput | Load Time | Read Throughput | CRC32C Stream Rate* | Top-K Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLAT** | 1,000 | 520,040 B (0.50 MB) | 520.0 B | 6.54 ms | 75.81 MB/s | 3.65 ms | 135.97 MB/s | 472.01 MB/s | **100%** |
| **FLAT** | 10,000 | 5,200,040 B (4.96 MB) | 520.0 B | 24.31 ms | 204.04 MB/s | 9.30 ms | **533.04 MB/s** | 1,776.01 MB/s | **100%** |
| **HNSW** | 1,000 | 638,336 B (0.61 MB) | 638.3 B | 8.84 ms | 68.89 MB/s | 3.24 ms | 187.93 MB/s | 784.49 MB/s | **100%** |
| **HNSW** | 10,000 | 6,357,070 B (6.06 MB) | 635.7 B | 27.57 ms | 219.92 MB/s | 26.31 ms | **230.41 MB/s** | 2,392.40 MB/s | **100%** |

*Notes: HNSW Configuration: $M=16, M_0=32, efConstruction=200, defaultEfSearch=50$. Ground truth behavioral equivalence verified with float tolerance $10^{-5}$. CRC32C stream rate reflects the measured throughput through Java's CRC32C streaming digest pipeline under this specific benchmark configuration, rather than an isolated hardware limit.*

#### Engineering Trade-Off Analysis

1. **Storage Footprint**:
   - **FLAT Index**: Pure raw storage efficiency ($520.0$ bytes/vector: 512 bytes for 128 floats + 8 bytes for external ID). Metadata and headers account for only 40 bytes overhead.
   - **HNSW Index**: Adds ~115.7 bytes/vector overhead to store the multi-layer navigable small-world graph topology (node levels, degree counts, and directed neighbor adjacency lists bounded by $M_0=32, M=16$).
2. **Deserialization Latency**:
   - Direct buffer restoration restores 10,000 128-dimensional Flat vectors in **9.30 ms** (533 MB/s).
   - Verbatim $O(E)$ HNSW topology reconstruction restores 10,000 nodes across all graph layers in **26.31 ms** (230 MB/s), restoring the persisted topology directly and avoiding graph reconstruction during load (for reference, index build took ~430 ms under this benchmark configuration).

---

## 🔬 Rigorous JMH Benchmark Suite (`nanovector-benchmark`)

NanoVector includes an isolated benchmarking module (`nanovector-benchmark`) powered by **JMH 1.37 (Java Microbenchmark Harness)** to empirically evaluate performance hypotheses without confounding factors (JIT compilation noise, GC pauses, dead-code elimination, or pre-assumed speedups).

All benchmarks run with Compiler Blackholes, JIT warmup cycles (2 iterations $\times$ 500 ms), measurement iterations (3 iterations $\times$ 500 ms), and isolated process forks on JDK 25 (`--release 21`, `--add-modules jdk.incubator.vector`).

### 1. Distance Kernel Factorial Matrix (Scalar vs SIMD)

Evaluated across 36 factorial configurations: 3 metrics (`EUCLIDEAN`, `COSINE`, `DOT_PRODUCT`) $\times$ 6 dimensions (32, 64, 128, 384, 768, 1536) $\times$ 2 engines (Scalar vs SIMD):

- **Peak Speedup Range**: SIMD acceleration delivers between **~2.7x** ($D=32$) and **~7.0x–7.35x** ($D=384$), tapering to **~5.3x–6.1x** at higher dimensions ($D=1536$).
- **Experimental Control**: Input vectors are pre-generated with fixed seeds outside timed regions; cosine normalization is performed strictly in `@Setup`.

### 2. End-to-End Search Throughput & Amdahl's Law Evaluation

Measured on synthetic datasets ($D=128$, `EUCLIDEAN`, $k=10$, $efSearch=50$, 128 queries cycled via Blackhole):

| Index Architecture | Vector Count ($N$) | Scalar Throughput | SIMD Throughput | SIMD Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **FlatIndex** | 1,000 | 10,177 ops/s | 51,132 ops/s | **5.02x** |
| **FlatIndex** | 10,000 | 1,038 ops/s | 3,092 ops/s | **2.98x** |
| **HnswIndex** | 1,000 | 9,131 ops/s | 18,321 ops/s | **2.01x** |
| **HnswIndex** | 10,000 | 3,932 ops/s | 6,354 ops/s | **1.62x** |

> [!NOTE]
> **Amdahl's Law in Action**:
> - FlatIndex có mức phụ thuộc cao hơn vào distance computation, thể hiện qua SIMD speedup 5.02× ở $N=1{,}000$ và 2.98× ở $N=10{,}000$.
> - Trong HnswIndex, ngoài tính khoảng cách còn có chi phí duyệt đồ thị (truy xuất láng giềng, bitset/epoch set, heap candidates), khiến tốc độ tăng tốc end-to-end chỉ đạt **1.62x–2.01x** dù raw distance kernel tăng tốc hơn 5x.
> - **Empirical scaling**: Flat throughput giảm khoảng 9.8× khi $N$ tăng 10×, trong khi HNSW giảm khoảng 2.3× trong workload này.

### 3. HNSW Pareto Frontier: Recall@10 vs Latency / Throughput

Swept across beam search widths ($efSearch \in \{10, 20, 50, 100, 200\}$) on $N=10{,}000$, $D=128$, `EUCLIDEAN`, $k=10$, measured against the exact `FlatIndex` reference oracle:

| `efSearch` | Empirical Recall@10 | Scalar QPS | SIMD QPS | SIMD Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **10** | 29.38% | 12,106 ops/s | 24,632 ops/s | **2.03x** |
| **20** | 44.53% | 7,577 ops/s | 15,657 ops/s | **2.07x** |
| **50** | 68.05% | 3,941 ops/s | 7,274 ops/s | **1.85x** |
| **100** | 85.47% | 2,206 ops/s | 3,734 ops/s | **1.69x** |
| **200** | 96.25% | 1,184 ops/s | 2,062 ops/s | **1.74x** |

- **Recall Invariance**: Scalar and SIMD produced identical measured Recall@10 across all tested `efSearch` values under this benchmark configuration.
- **Pareto Trade-off**: Increasing `efSearch` from 10 to 200 lifts Recall@10 from **29.38%** to **96.25%**, with a corresponding ~12x reduction in search throughput (from 24.6K ops/s down to 2.0K ops/s).

### 4. Allocation & GC Profiling (`-prof gc`)

Empirical testing of heap allocation behavior using JMH's normalized allocation profiler (`·gc.alloc.rate.norm`):

| Search Operation | Dataset Size ($N$) | Normalized Allocation (`B/op`) | Empirical Analysis |
| :--- | :---: | :---: | :--- |
| `baselineRawDistance` | 1K & 10K | **$\approx 10^{-4}$ B/op** | Effectively 0 B/op at the JMH measurement resolution; no meaningful heap allocation was observed in the raw primitive distance kernel. |
| `searchFlat` | 1,000 | **560.29 B/op** | Allocation remained approximately constant as $N$ increased from 1K to 10K under fixed $K$ (~560 B vs ~564 B; $N$-element distance scan generates zero allocations). |
| `searchFlat` | 10,000 | **564.36 B/op** | |
| `searchHnsw` ($ef=10$) | 10,000 | **2,074.14 B/op** | **Disproving zero-allocation for HNSW search**: Graph traversal allocates candidate nodes in dynamic priority queues and instantiates `SearchResult` records, scaling with beam width. |
| `searchHnsw` ($ef=50$) | 10,000 | **8,969.04 B/op** | |
| `searchHnsw` ($ef=100$) | 10,000 | **15,401.95 B/op** | |

### 5. Graph Topology Divergence & Construction Acceleration

Comparing Scalar-constructed vs SIMD-constructed graphs ($N=1{,}000$, $D=128$, `EUCLIDEAN`, Seed=42):

- **Topology Consistency**: No topology divergence was observed under this benchmark configuration:
  - Entry point: identical (Node 571)
  - Max level: identical (Level 2)
  - Total edges: 28,794 (Scalar) vs 28,794 (SIMD)
  - Shared edges: **28,794 (Edge Jaccard Similarity: 100.0000%)**
  - Identical nodes ratio: **100.00%**
  - Search Top-10 overlap across 100 queries: **100.00%**
- **Construction Throughput**: Under this benchmark configuration, SIMD distance acceleration sped up HNSW index construction by **~3.00×** (from ~704 ms to ~235 ms per 1,000 vectors).

> [!IMPORTANT]
> **Summary Takeaway**: Raw SIMD distance đạt mức tăng tốc lớn, nhưng end-to-end HNSW chỉ đạt 1.62–2.01× ở workload được đo, trong khi JMH cho thấy HNSW search vẫn phát sinh allocation đáng kể.

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

This runs all **215 unit and integration tests** (Core: 146, Persistence: 58, Benchmark: 11; 0 failures, 0 errors, 0 skipped) across Linux and Windows CI.

### Run Performance Benchmarks

To run the JMH benchmark suite packaged in `benchmarks.jar`:
```powershell
# Run distance kernel benchmarks across metrics and dimensions
java --add-modules jdk.incubator.vector -jar nanovector-benchmark/target/benchmarks.jar ScalarVsSimdDistanceBenchmark

# Run Flat vs HNSW search throughput benchmarks
java --add-modules jdk.incubator.vector -jar nanovector-benchmark/target/benchmarks.jar FlatVsHnswSearchBenchmark

# Sweep HNSW Pareto frontier (Recall vs Latency)
java --add-modules jdk.incubator.vector -jar nanovector-benchmark/target/benchmarks.jar EfSearchRecallLatencyBenchmark

# Profile heap allocation and GC rates
java --add-modules jdk.incubator.vector -jar nanovector-benchmark/target/benchmarks.jar SearchAllocationBenchmark -prof gc

# Run graph topology divergence and construction benchmarks
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar com.nanovector.benchmark.topology.GraphTopologyDivergenceBenchmark
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
  - [x] Zero-allocation distance evaluation hot path (eliminated intermediate array copying during node distance calculations).
  - [x] Graph invariant verification (Degree $\le M/M_0$, BFS connectivity, 100% bidirectional symmetry, seed determinism).
  - [x] Empirical Recall@10 verification against `FlatIndex` Oracle.
  - [x] Microbenchmark harness (`HnswBenchmark` measuring latency percentiles, throughput, and heap allocation).
  - [x] Cross-platform GitHub Actions CI (Ubuntu + Windows).
- [x] **v0.3 (Phase 3)**: Experimental SIMD acceleration via Java Vector API (`jdk.incubator.vector`) (146 unit tests):
  - [x] Incubator module configuration in Maven compiler, surefire, and GitHub Actions CI.
  - [x] Vectorized distance engines (`VectorEuclideanDistance`, `VectorCosineDistance`, `VectorDotProductDistance`).
  - [x] Hardware-adaptive lane sizing (`FloatVector.SPECIES_PREFERRED`) and scalar tail loop.
  - [x] Comprehensive scalar vs SIMD equivalence test suite across dimensions and metrics.
  - [x] Integration into `FlatIndex` and `HnswIndex` (default SIMD for HNSW, exact scalar reference oracle for Flat).
  - [x] Comprehensive benchmark suite (`SimdBenchmark`) demonstrating up to 5.55x raw distance speedup, 3.39x brute-force scan speedup, and 2.32x HNSW search speedup.
- [x] **v0.4 (Phase 4)**: Binary persistence (`.nvec` NVEC v1 Format) (58 tests in persistence module, 204 total):
  - [x] Binary format specification (`FORMAT_SPEC_V1.md`) with Little-Endian portability, 32B header, 28B HNSW metadata block, and CRC32C footer.
  - [x] `VectorDataView` read-only contract and `IndexRestorer` controlled internal bridge.
  - [x] Bounded-memory chunked I/O serializer (`NvecWriter`) with streaming CRC32C, atomic `.tmp` replace, and pre-write invariant checks.
  - [x] 8-step defense-in-depth deserializer (`NvecReader`) with pre-allocation bounds checks preventing OOM and forward-compatible metadata parsing.
  - [x] Direct verbatim $O(E)$ HNSW topology reconstruction without heuristic re-clustering or graph rebuild.
  - [x] End-to-end integration and round-trip persistence benchmark (`PersistenceBenchmark`).
- [x] **v0.5 (Phase 5)**: Rigorous JMH benchmark suite (`nanovector-benchmark`) (11 benchmark tests, 215 total):
  - [x] JMH harness setup with Maven shade plugin (`benchmarks.jar`), compiler blackholes, and zero coupling with persistence.
  - [x] Comprehensive 36-case distance kernel benchmark matrix (`ScalarVsSimdDistanceBenchmark`).
  - [x] Flat vs HNSW search benchmark evaluating Amdahl's Law and scale properties (`FlatVsHnswSearchBenchmark`).
  - [x] HNSW Pareto frontier exploration sweeping Recall@10 vs QPS across $efSearch$ (`EfSearchRecallLatencyBenchmark`).
  - [x] Heap allocation and GC rate profiling benchmark with `-prof gc` (`SearchAllocationBenchmark`).
  - [x] HNSW topology divergence analysis and index construction throughput benchmark (`GraphTopologyDivergenceBenchmark`).
- [ ] **v0.6 (Phase 6)**: Scale & memory experiments (quantization study, off-heap MemorySegment evaluation).
- [ ] **v0.7 (Phase 7)**: Standalone CLI & Spring Boot REST API.


