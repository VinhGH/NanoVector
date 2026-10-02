# NanoVector

An experimental, high-performance in-memory vector similarity search engine written in pure Java (Java 25+ / LTS), designed to explore cache-friendly memory layouts, vector indexing algorithms (Flat baseline, HNSW), Foreign Function & Memory (FFM) off-heap layouts, and systems performance engineering.

---

## 🎯 Architecture & Modules

NanoVector is organized as a modular, decoupled multi-module Maven project:

```text
NanoVector
│
├── nanovector-core          # Vector storage, SIMD distance kernels, HNSW, SQ8 quantization, Off-Heap FFM layouts
├── nanovector-persistence   # NVEC v1 streaming binary format with hardware-accelerated CRC32C and atomic file replacement
├── nanovector-benchmark     # Comprehensive JMH benchmarks, recall sweeps, memory profiling, and allocation diagnostics
├── nanovector-cli           # Standalone command-line interface tool (Picocli) for index creation, inspection, and querying
└── nanovector-server        # Production-ready Spring Boot 3.4 REST microservice, OpenAPI/Swagger UI, and concurrent session manager
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

## 🔬 Scale & Memory Characterization (Phase 6A)

Phase 6A conducts systematic empirical profiling of NanoVector across dataset scales ($N \in \{1\text{K}, 10\text{K}, 50\text{K}, 100\text{K}\}$ at dimension $D=128$, `EUCLIDEAN`, $k=10$, $efSearch=50$, SIMD-accelerated).

### 1. Memory Footprint: Analytical Structural Model vs Measured Heap Delta

To prepare a rigorous baseline for quantization (Phase 6B), NanoVector decouples **Analytical Structural Cost** (HotSpot 64-bit with Compressed OOPs) from **Measured Heap Delta** (`Runtime.getRuntime()` with 4-pass GC stabilization):

| Scale ($N$) | Index Type | Raw Payload | Structural Memory | Structural B/vec | Measured Heap Delta | Measured B/vec | Memory Amplification |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | FlatIndex | 0.49 MiB | 0.58 MiB | 609.4 B | 0.60 MiB | 626.8 B | **1.19x** (struct) / **1.22x** (meas) |
| **1,000** | HnswIndex | 0.49 MiB | 0.77 MiB | 805.2 B | 0.79 MiB | 832.0 B | **1.57x** (struct) / **1.63x** (meas) |
| **10,000** | FlatIndex | 4.88 MiB | 5.75 MiB | 603.2 B | 5.96 MiB | 624.9 B | **1.18x** (struct) / **1.22x** (meas) |
| **10,000** | HnswIndex | 4.88 MiB | 7.48 MiB | 784.8 B | 7.74 MiB | 811.8 B | **1.53x** (struct) / **1.59x** (meas) |
| **50,000** | FlatIndex | 24.41 MiB | 28.73 MiB | 602.6 B | 29.83 MiB | 625.5 B | **1.18x** (struct) / **1.22x** (meas) |
| **50,000** | HnswIndex | 24.41 MiB | 37.39 MiB | 784.2 B | 38.31 MiB | 803.4 B | **1.53x** (struct) / **1.57x** (meas) |
| **100,000** | FlatIndex | 48.83 MiB | 57.46 MiB | 602.5 B | 59.62 MiB | 625.2 B | **1.18x** (struct) / **1.22x** (meas) |
| **100,000** | HnswIndex | 48.83 MiB | 74.78 MiB | 784.1 B | 76.29 MiB | 800.0 B | **1.53x** (struct) / **1.56x** (meas) |

> [!NOTE]
> **Key Architectural Takeaways**:
> - **FlatIndex Overhead**: Consumes ~602.5 B/vec vs 512 B raw payload (amplification **1.18x**), dominated by the 8B external ID buffer and ~82.5 B/vec `HashMap<Long, Integer>` index overhead.
> - **HNSW Graph Overhead**: Graph topology adds ~181.6 B/vec structural overhead (multi-layer `HnswNode` pointers, neighbor adjacency arrays, and `EpochVisitedSet`), bringing total footprint to ~784.1 B/vec (amplification **1.53x**).
> - **Model Accuracy**: Analytical structural estimates match empirical heap deltas within **2.0%** at 100K scale under the tested JVM configuration.
> - **Analytical Projection for Phase 6B (Scalar Quantization SQ8)**: In FP32, raw vectors account for 65.3% of HNSW memory, while graph topology accounts for 23.2%. Under an analytical SQ8 projection (1 byte/dimension), 4x vector compression (512B $\to$ 128B) would invert this ratio, reducing total projected HNSW memory to ~38.16 MiB where graph topology becomes the dominant memory consumer (~45.4% of total index memory).

### 2. Search Latency & Throughput Scaling Across 100x Scale

Empirical latency distribution and query throughput ($D=128$, `EUCLIDEAN`, $k=10$, $efSearch=50$, SIMD-accelerated):

| Scale ($N$) | Index Type | Working Set | QPS (ops/s) | Mean Latency | P50 Latency | P90 Latency | P99 Latency | Empirical Recall@10 | Speedup |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | FlatIndex | 0.49 MiB | 20,180.7 | 49.16 μs | 41.30 μs | 83.40 μs | 146.70 μs | 100.00% | 1.00x |
| **1,000** | HnswIndex | 0.49 MiB | 7,157.9 | 139.31 μs | 83.00 μs | 229.60 μs | 518.50 μs | 99.38% | 0.35x |
| **10,000** | FlatIndex | 4.88 MiB | 1,802.4 | 550.22 μs | 519.20 μs | 706.70 μs | 1,163.00 μs | 100.00% | 1.00x |
| **10,000** | HnswIndex | 4.88 MiB | 2,268.0 | 434.67 μs | 396.60 μs | 607.40 μs | 1,167.70 μs | 68.05% | **1.27x** |
| **50,000** | FlatIndex | 24.41 MiB | 575.8 | 1,734.12 μs | 1,666.50 μs | 2,046.70 μs | 2,482.20 μs | 100.00% | 1.00x |
| **50,000** | HnswIndex | 24.41 MiB | 3,817.0 | 261.51 μs | 252.90 μs | 310.20 μs | 410.80 μs | 39.45% | **6.63x** |
| **100,000** | FlatIndex | 48.83 MiB | 267.0 | 3,741.09 μs | 3,598.30 μs | 4,484.00 μs | 5,105.40 μs | 100.00% | 1.00x |
| **100,000** | HnswIndex | 48.83 MiB | 2,953.1 | 337.62 μs | 321.50 μs | 419.50 μs | 662.60 μs | 27.42% | **11.08x** |

> [!NOTE]
> **Performance Observations**:
> - **Crossover Point**: At $N=1{,}000$, FlatIndex brute force is faster than HNSW ($0.35\times$ throughput) because scanning 1,000 contiguous vectors with SIMD incurs lower constant overhead than navigating HNSW's priority queues and visited sets. At $N=10{,}000$, HNSW overtakes Flat ($1.27\times$), expanding to **6.63x** at 50K and **11.08x** at 100K.
> - **Speedup vs Recall Interdependence**: At $N=100\text{K}$, HnswIndex achieves **11.08x higher throughput** than FlatIndex (2,953 ops/s vs 267 ops/s) under the tested $efSearch=50$ configuration, though at a lower Recall@10 of 27.42% (post-patch; pre-patch was 28.20%, representing a minor -0.78% trade-off to strictly protect Layer-0 graph connectivity). In approximate nearest neighbor search, throughput speedup cannot be evaluated independently from the corresponding recall level.
> - **Empirical Scaling**: Across the 100-fold scale increase ($1\text{K} \to 100\text{K}$), FlatIndex throughput collapsed $75.6\times$ (from 20.2K QPS to 267 QPS), whereas HnswIndex latency remained within 139–338 μs.
> - **Sequential Scan vs Graph Traversal**: Each FlatIndex query sequentially scans the entire contiguous vector payload (from 0.49 MiB at 1K up to 48.83 MiB at 100K). At 100K scale, this payload exceeds typical CPU L3 cache capacities, which can necessitate DRAM fetches for each query and constrain throughput; whereas HnswIndex evaluates only a small subset of vectors along the traversal path (cache residency and bus traffic were not measured directly via hardware performance counters).
> - **Search Budget Scaling**: At a *fixed* $efSearch=50$, empirical Recall@10 decreases from 99.38% at 1K down to 27.42% at 100K. This confirms that a fixed candidate budget cannot maintain retrieval quality as the graph search space expands 100-fold; achieving target recall at scale requires scaling the search budget ($efSearch$), though determining the exact scaling relationship (e.g. logarithmic vs polynomial) requires further multi-parameter Pareto sweeps.

### 3. HNSW Construction Scaling & Multi-Layer Topology Breakdown

Empirical construction performance and graph topology properties across scales ($D=128$, `EUCLIDEAN`, $M=16, M_0=32, efConstruction=200$, SIMD-accelerated):

| Scale ($N$) | Raw Payload | Cumulative Build Time | Insertion Throughput | Latency / Vector | Max Graph Level | Layer 0 Isolated Nodes | Connected Components | Total Edges |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 0.49 MiB | 491 ms | 2,034.8 vec/s | 491.44 μs | Level 2 | **0** | **1** (100% BFS reachable) | 28,796 |
| **10,000** | 4.88 MiB | 5,545 ms | 1,803.3 vec/s | 554.55 μs | Level 3 | **0** | **1** (100% BFS reachable) | 281,452 |
| **50,000** | 24.41 MiB | 47,836 ms | 1,045.2 vec/s | 956.72 μs | Level 4 | **0** | **1** (100% BFS reachable) | 1,292,374 |
| **100,000** | 48.83 MiB | 117,858 ms (~1.96 min) | 848.5 vec/s | 1,178.59 μs | Level 6 | **0** | **1** (100% BFS reachable) | 2,475,330 |

#### Multi-Layer Graph Topology at Scale $N=100{,}000$ ($M=16, M_0=32$):

| Layer | Node Count | % of Total | Min Degree | Avg Degree | Max Degree | Max Allowed | Invariant Verification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Layer 0** | 100,000 | 100.00% | **1** | 23.89 | 32 | 32 | Degree $\le M_0$; 100% BFS connected (1 component, 0 isolated nodes) |
| **Layer 1** | 6,188 | 6.19% | 0* | 12.94 | 16 | 16 | Degree $\le M$ strictly enforced; ~1/16 decay ratio |
| **Layer 2** | 387 | 0.39% | 5 | 14.33 | 16 | 16 | Degree $\le M$ strictly enforced; ~1/16 decay ratio |
| **Layer 3** | 26 | 0.03% | 5 | 14.77 | 16 | 16 | Degree $\le M$ strictly enforced; ~1/16 decay ratio |
| **Layer 4** | 3 | <0.01% | 2 | 2.00 | 2 | 16 | Bounded degree |
| **Layer 5** | 1 | <0.01% | 0* | 0.00 | 0 | 16 | Single entry candidate (degree 0: no peers at this level) |
| **Layer 6** | 1 | <0.01% | 0* | 0.00 | 0 | 16 | Global Entry Point (degree 0: single top-level node) |

*\*Note on higher layers with degree 0: When a top layer contains exactly 1 node ($n=1$), no peer nodes exist at that level to form edges (self-loops are prohibited). Search traversal immediately descends to populated lower layers.*

> [!TIP]
> **Topology Insights**:
> - **Exponential Layer Decay**: Node count drops by an empirical factor of $\approx 1/16$ per layer ($100{,}000 \to 6{,}188 \to 387 \to 26 \to 3 \to 1 \to 1$), adhering closely to the theoretical level multiplier $m_L = 1/\ln(M)$.
> - **Degree Invariant Enforcement**: All layers strictly respect their degree capping ($M_0=32$ for Layer 0, $M=16$ for higher layers).
> - **True Graph Connectivity vs Degree Invariant**: Differentiates between two separate structural guarantees:
>   - *Invariant A (Zero Isolated Nodes)*: Verified $\min(\text{degree}) \ge 1$ across all nodes at Layer 0 (0 isolated nodes).
>   - *Invariant B (Full Graph Connectivity)*: Verified via exhaustive Breadth-First Search (BFS) from node 0, confirming exactly **1 connected component** with $100\%$ node reachability (100,000 / 100,000 nodes reachable at 100K scale).
> - **Throughput Decay Profile**: Insertion throughput gradually decays from ~2,035 vec/s down to ~849 vec/s as graph depth expands from 2 to 6 layers, requiring deeper beam search traversal ($efConstruction=200$) on insertion.


---

## 🗜️ Scalar Quantization (SQ8) Study (Phase 6B)

Phase 6B investigates the fundamental systems question:
> **"Can we reduce vector-storage memory by approximately 4× while preserving acceptable Recall@10 and obtaining useful distance-computation performance?"**

To isolate quantization distortion from graph-routing heuristics, Phase 6B **strictly excludes HNSW**, benchmarking the quantized flat index (`QuantizedFlatIndex`) directly against the exact FP32 ground truth oracle (`FlatIndex`) across $N \in \{1\text{K}, 10\text{K}, 50\text{K}, 100\text{K}\}$ at dimension $D=128$ under `EUCLIDEAN` distance ($k=10$, 128 queries, seeds: 42L data, 12345L query). Within this brute-force FlatIndex comparison, the observed Recall difference isolates the ranking impact of SQ8 distance approximation, without HNSW routing effects.

### 1. Mathematical Representation & Contiguous Storage

- **Granularity & Mapping**: Per-vector asymmetric affine quantization mapping continuous $v_j \in [v_{\min}, v_{\max}]$ to unsigned 8-bit integers $[0, 255]$:
  $$\text{scale} = \frac{v_{\max} - v_{\min}}{255.0f}, \quad q_j = \text{clamp}\left( \text{round}\left( \frac{v_j - v_{\min}}{\text{scale}} \right), 0, 255 \right)$$
  *(degenerate zero-range where $v_{\max} == v_{\min}$ sets $\text{scale} = 0.0f$, $q_j = 0$, reconstructing $v_{\min}$ exactly)*
- **Storage Representation**: Unsigned 8-bit values stored in Java primitive `byte[]` buffers (`(byte) (q & 0xFF)`), retrieved via `raw & 0xFF`.
- **Per-Vector Metadata**: `float min` (4 bytes) + `float scale` (4 bytes) = 8 bytes metadata per vector.
- **Payload at $D=128$**: $128 \times 1\text{ B} + 8\text{ B} = 136\text{ bytes}$ per vector, yielding a **$3.76\times$ vector storage compression** compared to FP32 ($128 \times 4 = 512\text{ bytes}$).
- **Contiguous Primitive Storage**: `QuantizedVectorStorage` maintains flat primitive buffers (`byte[] quantizedBuffer`, `float[] minBuffer`, `float[] scaleBuffer`, `long[] externalIds`), ensuring distance evaluation avoids intermediate vector-array allocations.

### 2. Asymmetric Distance Computation (ADC) & Vector API SIMD

- **Asymmetric Distance Computation (ADC)**: The query vector $q$ remains in exact FP32 single-precision to avoid double quantization distortion:
  $$L_2^2(q, \hat{v}) = \sum_{j=0}^{D-1} \left( q_j - (v_{\min} + \text{unsigned}(p_j) \times \text{scale}) \right)^2$$
- **SIMD Acceleration (`jdk.incubator.vector`)**: `VectorQuantizedEuclideanDistance` uses `FloatVector.SPECIES_PREFERRED` (measured as 8 float lanes on the benchmark machine's AVX2 configuration), loading 8 bytes, widening unsigned bytes to integers, casting to `FloatVector`, reconstructing floats via FMA ($\text{laneMin} + v_{\text{float}} \times \text{laneScale}$), and accumulating squared differences using FMA instructions, with a scalar tail loop for dimension alignment.

### 3. Empirical Recall@10 Trade-Off ($D=128, k=10$)

Measured against the exact FP32 `FlatIndex` reference oracle across scales:

| Scale ($N$) | FP32 Flat Recall (Oracle) | SQ8 Scalar Recall | SQ8 SIMD Recall | Recall Loss | Scalar-SIMD Agreement |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 100.00% | **99.45%** | **99.45%** | **0.55%** | **100.00%** |
| **10,000** | 100.00% | **98.91%** | **98.91%** | **1.09%** | **100.00%** |
| **50,000** | 100.00% | **98.91%** | **98.91%** | **1.09%** | **100.00%** |
| **100,000** | 100.00% | **99.30%** | **99.30%** | **0.70%** | **100.00%** |

> [!NOTE]
> **Recall Observations**:
> - **Ranking Impact**: Under this benchmark configuration, SQ8 quantization produced limited Top-10 ranking disruption, with empirical Recall@10 remaining strictly bounded between **98.91%** and **99.45%** (peak recall loss $\le 1.09\%$).
> - **Non-Monotonic Recall Behavior**: Recall loss does not degrade monotonically with $N$ ($0.55\% \to 1.09\% \to 1.09\% \to 0.70\%$), reflecting local neighborhood density variations rather than cumulative distortion.
> - **Algorithmic Parity**: Scalar ADC and SIMD ADC produce **100.00% mutual top-$k$ agreement** across all tested queries and scales, proving that vectorized arithmetic widening and FMA accumulation preserve exact ranking order.

### 4. Memory Footprint across Scales: Three Memory Tiers

Measured via `MemoryFootprintProfiler` (structural HotSpot 64-bit model vs empirical 4-pass GC heap delta):

| Scale ($N$) | FP32 Raw MB | SQ8 Raw MB | FP32 Vec Storage | SQ8 Vec Storage | Vec Storage Red. | FP32 Struct MB | SQ8 Struct MB | Total Struct Red. | FP32 Heap Delta | SQ8 Heap Delta | Heap Delta Red. |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 0.49 MiB | 0.13 MiB | 512 B | 136 B | **3.76x** | 0.57 MiB | 0.21 MiB | **2.68x** | 0.57 MiB | 0.23 MiB | **2.42x** |
| **10,000** | 4.88 MiB | 1.30 MiB | 512 B | 136 B | **3.76x** | 5.71 MiB | 2.12 MiB | **2.69x** | 6.85 MiB | 2.90 MiB | **2.36x** |
| **50,000** | 24.41 MiB | 6.48 MiB | 512 B | 136 B | **3.76x** | 28.73 MiB | 10.80 MiB | **2.66x** | 30.35 MiB | 12.69 MiB | **2.39x** |
| **100,000** | 48.83 MiB | 12.97 MiB | 512 B | 136 B | **3.76x** | 57.46 MiB | 21.60 MiB | **2.66x** | 59.59 MiB | 24.38 MiB | **2.44x** |

- **Tier 1 (Vector Storage Payload)**: Achieves **$3.76\times$ reduction** ($512\text{ B} \to 136\text{ B}$ per vector, accounting for the 8-byte $min/scale$ metadata).
- **Tier 2 (Total Structural Index)**: Achieves **$2.66\times - 2.69\times$ reduction** ($57.46\text{ MiB} \to 21.60\text{ MiB}$ at 100K), constrained by the fixed auxiliary cost of `long[] externalIds` (8 B/vec) and `HashMap<Long, Integer>` (~82.5 B/vec).
- **Tier 3 (Measured JVM Heap Delta)**: Achieves **$2.36\times - 2.44\times$ reduction** in observed runtime heap delta ($59.59\text{ MiB} \to 24.38\text{ MiB}$ at 100K) under the benchmark methodology, reflecting JVM object/array overheads and allocator/GC behavior (an empirical runtime observation, not an exact retained-object footprint).

### 5. Search Latency Distribution & Throughput Dynamics

Measured on `FlatVsQuantizedFlatBenchmark` ($D=128, k=10$, 128 queries):

| Scale ($N$) | Configuration | Throughput (QPS) | Mean Latency | P50 Latency | P95 Latency | P99 Latency | Speedup vs FP32 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | FP32 Flat (Scalar) | 7,738.0 | 129.04 μs | 114.30 μs | 187.00 μs | 226.80 μs | 1.00x |
| | SQ8 Flat (Scalar ADC) | 4,733.7 | 210.84 μs | 129.80 μs | 632.00 μs | 1,048.30 μs | **0.61x** |
| | FP32 Flat (SIMD) | 15,342.6 | 64.87 μs | 55.30 μs | 139.00 μs | 228.20 μs | 1.00x |
| | SQ8 Flat (SIMD ADC) | 15,316.4 | 65.08 μs | 49.50 μs | 134.60 μs | 252.00 μs | **1.00x** |
| **10,000** | FP32 Flat (Scalar) | 976.3 | 1,022.50 μs | 1,012.90 μs | 1,114.00 μs | 1,260.60 μs | 1.00x |
| | SQ8 Flat (Scalar ADC) | 893.5 | 1,118.28 μs | 1,087.50 μs | 1,264.20 μs | 1,992.50 μs | **0.91x** |
| | FP32 Flat (SIMD) | 2,451.9 | 406.37 μs | 381.30 μs | 629.60 μs | 846.90 μs | 1.00x |
| | SQ8 Flat (SIMD ADC) | 3,536.5 | 282.40 μs | 277.70 μs | 314.00 μs | 369.10 μs | **1.44x** |
| **50,000** | FP32 Flat (Scalar) | 185.7 | 5,382.60 μs | 5,298.20 μs | 6,017.00 μs | 7,070.30 μs | 1.00x |
| | SQ8 Flat (Scalar ADC) | 177.8 | 5,621.05 μs | 5,493.20 μs | 6,349.20 μs | 7,275.40 μs | **0.96x** |
| | FP32 Flat (SIMD) | 394.8 | 2,530.23 μs | 2,436.90 μs | 3,496.20 μs | 3,926.20 μs | 1.00x |
| | SQ8 Flat (SIMD ADC) | 693.1 | 1,440.61 μs | 1,409.10 μs | 1,656.30 μs | 1,914.60 μs | **1.76x** |
| **100,000** | FP32 Flat (Scalar) | 94.0 | 10,630.95 μs | 10,495.80 μs | 11,516.10 μs | 12,788.90 μs | 1.00x |
| | SQ8 Flat (Scalar ADC) | 88.7 | 11,274.68 μs | 11,054.50 μs | 12,468.10 μs | 15,627.90 μs | **0.94x** |
| | FP32 Flat (SIMD) | 205.1 | 4,873.35 μs | 4,784.00 μs | 6,111.70 μs | 7,249.00 μs | 1.00x |
| | SQ8 Flat (SIMD ADC) | 314.9 | 3,172.40 μs | 2,893.60 μs | 4,505.90 μs | 6,011.10 μs | **1.54x** |

### 6. Systems Engineering Takeaways

1. **Scalar Arithmetic Overhead (CPU-Bound Bottleneck)**:
   - In pure scalar execution, SQ8 Flat is consistently slower than or on par with FP32 Flat ($0.61\times - 0.96\times$).
   - *Root Cause*: Sequential scalar ADC requires additional per-coordinate ALU operations (byte sign mask `& 0xFF`, integer-to-float widening, scale multiplication, and min offset addition) before computing $(q_j - \hat{v}_j)^2$. Memory footprint reduction does not inherently yield CPU execution speedup when decoding instructions dominate execution time.
2. **Possible Bottleneck Transition**:
   - At $N=1\text{K}$, the FP32 and SQ8 vector payloads are relatively small, and SQ8 SIMD remains on par or slightly slower than FP32 SIMD ($0.85\times - 1.00\times$).
   - As $N$ increases ($10\text{K} \to 100\text{K}$), SQ8 SIMD consistently outperforms FP32 SIMD (**$1.42\times - 1.76\times$ speedup**).
   - This behavior is consistent with reduced memory traffic becoming increasingly valuable at larger working-set sizes, but cache residency and DRAM bandwidth were not directly measured via hardware performance counters.
3. **Synthesis Verdict**:
   > *"Under the tested configuration, asymmetric per-vector SQ8 substantially reduces vector-storage ($3.76\times$) and measured heap usage ($2.36\times - 2.44\times$) while maintaining high measured Recall@10 ($98.91\% - 99.45\%$, maximum recall loss $\le 1.09\%$). Its performance benefit depends on the execution path and dataset scale: scalar ADC does not consistently outperform FP32, whereas SIMD ADC becomes advantageous at larger scales ($1.42\times - 1.76\times$)."*

---

## ⚡ Quantized HNSW Systems Study (Phase 6C)

Phase 6C integrates 8-bit asymmetric per-vector Scalar Quantization (SQ8) and Asymmetric Distance Computation (ADC) directly into the HNSW routing and graph construction engines (`QuantizedHnswIndex`), answering three fundamental systems questions:
> 1. **Graph Construction Under Quantization**: Does building an HNSW graph using approximate SQ8 SIMD ADC distances degrade the graph topology, violate structural connectivity invariants, or damage small-world navigability?
> 2. **Real-World Memory Footprint & Structural Validation**: Does the analytical structural projection from Phase 6B ($\approx 38.92\text{ MiB}$ at $100\text{K}$) hold under empirical JVM heap delta measurement? Does graph topology become the dominant memory consumer when vector storage is compressed?
> 3. **Pareto Frontier & Two-Phase Search Recovery**: How does the Recall vs Throughput trade-off shift across dynamic beam widths ($efSearch \in \{10, \dots, 400\}$)? Can a Two-Phase search (SQ8 graph exploration followed by exact FP32 re-ranking) recover the quantization-induced recall loss?

Baseline Workload: Uniform synthetic vectors, $D = 128$, Metric = Squared Euclidean ($L_2^2$), $k = 10$, 128 test queries, seeds `42L` (dataset) and `12345L` (queries). Reference Oracle: Full-precision FP32 `FlatIndex` ($O(N)$ sequential scan, 100% recall by definition).

### 1. Architectural Design & Separation of Concerns

`QuantizedHnswIndex` implements the `VectorIndex` contract by orchestrating three cleanly decoupled layers:

```text
QuantizedHnswIndex (VectorIndex contract: insert, searchKnn, searchKnnWithRerank)
    │
    ├── QuantizedVectorStorage (136 B/vec: byte[] vectors, float[] mins, float[] scales, long[] externalIds)
    │
    ├── HnswGraph (Decoupled topology: HnswNode[], LevelGenerator, EpochVisitedSet)
    │       │
    │       └── NeighborSelector (Algorithm 4 heuristic with fallback)
    │
    └── QuantizedEuclideanDistance (SIMD ADC distance kernel via jdk.incubator.vector)
```

- **Decoupled Functional Evaluators**: Pure graph components (`HnswGraph`, `NeighborSelector`) do not touch vector data. Distances are evaluated via two functional interfaces:
  - `DistanceToQuery`: Evaluates distance from stored quantized nodes directly to an external query vector using SIMD ADC against the primitive byte buffer, with zero object allocations.
  - `NodeDistanceEvaluator`: Evaluates distance between two stored nodes during graph construction. In `QuantizedHnswIndex`, node A is dequantized into a single pre-allocated thread-safe scratch buffer (`float[] nodeEvalBuffer`), and node B is evaluated via ADC against that buffer.
- **Construction Strategies Evaluated**:
  - **FP32 HNSW (Reference)**: Exact FP32 Euclidean (`float[]`) for build and search (Reference Oracle graph topology).
  - **Hybrid SQ8 HNSW (Strategy A)**: Built with exact FP32 Euclidean (`HnswIndex.fromFp32`), searched with SQ8 SIMD ADC. Reuses FP32 reference graph; **isolates search quantization error from routing error**.
  - **Pure SQ8 HNSW (Strategy B)**: Built and searched end-to-end with SQ8 SIMD ADC (`byte[]` + scratch buffer dequantization). End-to-end quantized system with minimal build memory.

### 2. Graph Construction Strategies & Topology Divergence ($D=128, k=10, efSearch=50, efConstruction=200$)

Evaluated using `HnswConstructionStrategyBenchmark` across scales $N \in \{1\text{K}, 10\text{K}, 50\text{K}, 100\text{K}\}$:

#### Construction Performance & Build Throughput

| Scale $N$ | FP32 Build Time | FP32 Throughput | Pure SQ8 Build Time | Pure SQ8 Throughput | Build Time Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 610 ms | 1,636.8 vec/s | 1,032 ms | 968.6 vec/s | **1.69x** |
| **10,000** | 8,219 ms | 1,216.6 vec/s | 15,368 ms | 650.7 vec/s | **1.87x** |
| **50,000** | 70,195 ms | 712.3 vec/s | 105,141 ms | 475.6 vec/s | **1.50x** |
| **100,000** | 147,568 ms | 677.7 vec/s | 239,510 ms | 417.5 vec/s | **1.62x** |

> [!NOTE]
> **Construction Slowdown Dynamics**:
> - **Observed**: Pure SQ8 HNSW construction was $1.50\times - 1.87\times$ slower than FP32 HNSW across the tested scales.
> - **Implementation-Level Explanation**: The additional cost is consistent with the current implementation's ADC distance calculation and neighbor-selection/dequantization work: candidate exploration computes affine min/scale transformations on each vector pair, and neighbor heuristic pruning (`NeighborSelector`) dequantizes node vectors into a scratch buffer before evaluating diverse connections. FP32 HNSW evaluates direct contiguous float arrays without affine arithmetic.

#### Graph Topological Divergence & Invariant Health

| Scale $N$ | Layer-0 Jaccard Sim | FP32 Total Edges | Pure SQ8 Edges | Max Deg L0 | Isolated L0 | Components L0 | Topology Health |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | **96.07%** | 28,796 | 28,784 | 32 | 0 | 1 | **PASS** |
| **10,000** | **92.22%** | 281,452 | 281,518 | 32 | 0 | 1 | **PASS** |
| **50,000** | **76.72%** | 1,292,374 | 1,292,362 | 32 | 0 | 1 | **PASS** |
| **100,000** | **60.24%** | 2,475,330 | 2,477,594 | 32 | 0 | 1 | **PASS** |

> [!IMPORTANT]
> **Graph Structural Health**: Across all scales, Pure SQ8 graphs exhibit **zero isolated nodes** and **exactly 1 connected component** on Layer 0 (100% BFS reachability), with degree bounds ($\le 32$ at Layer 0) strictly respected. Total edge counts match the FP32 reference graph within $0.09\%$. While individual neighbor choices diverge as scale expands (Layer-0 Jaccard similarity drops to $60.24\%$ at $100\text{K}$ because quantization noise disrupts close distance ties), connectivity and graph structural invariants remained uncompromised in these experiments.

#### Search Recall@10 Breakdown & Error Attribution

By comparing FP32 HNSW, Hybrid SQ8 HNSW (Strategy A), and Pure SQ8 HNSW (Strategy B) against the exact FP32 `FlatIndex` Ground Truth Oracle, we decompose the total recall loss into its exact physical causes:
$$\Delta_{\text{total}} = \text{Recall}_{FP32} - \text{Recall}_{Pure} = \underbrace{(\text{Recall}_{FP32} - \text{Recall}_{Hybrid})}_{\text{Quantization Distance Loss } (\Delta_{\text{dist}})} + \underbrace{(\text{Recall}_{Hybrid} - \text{Recall}_{Pure})}_{\text{Topology Divergence Loss } (\Delta_{\text{topo}})}$$

| Scale $N$ | FP32 Recall | Hybrid Recall | Pure SQ8 Recall | Total Recall Loss | Quant Distance Loss | Topology Loss (Attrib) | Hybrid-Pure Agreement |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 99.38% | 98.83% | 98.75% | **0.62%** | 0.55% | 0.08% | 99.61% |
| **10,000** | 68.05% | 67.73% | 67.34% | **0.70%** | 0.31% | 0.39% | 94.45% |
| **50,000** | 39.45% | 40.16% | 39.14% | **0.31%** | -0.70% | 1.02% | 68.67% |
| **100,000** | 27.42% | 27.50% | 26.48% | **0.94%** | -0.08% | 1.02% | 49.22% |

> [!TIP]
> **Recall Attribution & Fixed-Budget Search Scaling ($efSearch=50$)**:
> - **Attribution Focus**: The primary purpose of Tables 1–3 is to isolate the recall impact of SQ8 quantization against the FP32 baseline, rather than demonstrating optimized absolute HNSW recall at scale.
> - **Search Budget Context**: Both FP32 and SQ8 recall decrease as scale expands ($99.38\% \to 27.42\%$ for FP32; $98.75\% \to 26.48\%$ for Pure SQ8) because $efSearch=50$ is held constant while the search space expands 100-fold (as characterized in Phase 6A; higher recall at scale requires scaling $efSearch$, as shown in Section 4).
> - **Empirical Stability**: What is meaningful is that the **gap between FP32 and Pure SQ8 remains strictly bounded within $\le 0.94$ percentage points** across all scales ($0.62\% \to 0.70\% \to 0.31\% \to 0.94\%$). Quantization substantially changes local graph topology at larger scales (Layer-0 Jaccard drops to $60.24\%$ at $100\text{K}$), while the resulting routing quality remains relatively stable in these experiments.

### 3. Measured Scale Memory Characterization ($1\text{K} \to 100\text{K}$)

Measured using `MemoryFootprintProfiler` on HotSpot 64-bit JVM with Compressed OOPs, under 4-pass GC stabilization:

| Scale $N$ | Raw Vector Payload | FP32 Vector Storage | FP32 Graph Topology | FP32 Total Structural | FP32 Measured Heap | SQ8 Vector Storage | SQ8 Graph Topology | SQ8 Total Structural | SQ8 Measured Heap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 0.49 MiB | 0.57 MiB | 0.19 MiB | 0.76 MiB | 0.93 MiB | 0.21 MiB | 0.19 MiB | **0.40 MiB** | **0.45 MiB** |
| **10,000** | 4.88 MiB | 5.71 MiB | 1.86 MiB | 7.57 MiB | 10.62 MiB | 2.12 MiB | 1.86 MiB | **3.98 MiB** | **3.92 MiB** |
| **50,000** | 24.41 MiB | 28.73 MiB | 8.87 MiB | 37.60 MiB | 40.88 MiB | 10.80 MiB | 8.87 MiB | **19.67 MiB** | **21.26 MiB** |
| **100,000** | 48.83 MiB | 57.46 MiB | 17.32 MiB | 74.78 MiB | 77.29 MiB | 21.60 MiB | 17.33 MiB | **38.93 MiB** | **42.01 MiB** |

#### Memory Reduction Ratios & Topology Dominance:

| Scale $N$ | FP32 Structural | FP32 Measured Heap | SQ8 Structural | SQ8 Measured Heap | Storage Reduction | Total Struct Reduction | Heap Delta Reduction | SQ8 Topology Fraction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 797.8 B/v | 972.6 B/v | 421.8 B/v | 473.8 B/v | 2.68x | 1.89x | **2.05x** | **46.8%** |
| **10,000** | 793.7 B/v | 1,113.7 B/v | 417.7 B/v | 410.9 B/v | 2.69x | 1.90x | **2.71x** | **46.7%** |
| **50,000** | 788.5 B/v | 857.3 B/v | 412.5 B/v | 445.9 B/v | 2.66x | 1.91x | **1.92x** | **45.1%** |
| **100,000** | 784.1 B/v | 810.4 B/v | 408.2 B/v | 440.6 B/v | 2.66x | 1.92x | **1.84x** | **44.5%** |

```
Memory Footprint Distribution at Scale N = 100,000:

FP32 HNSW (74.78 MiB Structural):
[████████████████████████████████████████████████████████ 76.8% Storage ][██████████████ 23.2% Graph ]

Pure SQ8 HNSW (38.93 MiB Structural):
[████████████████████████████ 55.5% Storage ][███████████████████████ 44.5% Graph ]
```

> [!IMPORTANT]
> **Validation of the Phase 6B Structural Model & Topology Dominance**:
> 1. In Phase 6B, our structural model projected **$\approx 38.92\text{ MiB}$** at $100\text{K}$. The measured structural footprint is **$38.93\text{ MiB}$** ($99.97\%$ exact agreement).
> 2. The measured JVM heap delta at $100\text{K}$ is **$42.01\text{ MiB}$**, reducing heap consumption by **$35.28\text{ MiB}$** ($1.84\times$ reduction vs FP32's $77.29\text{ MiB}$).
> 3. **Topology Dominance (Remaining Memory Bottleneck After Vector Compression)**: In FP32 HNSW, vector storage is the dominant cost ($76.8\%$). When vector storage is compressed via SQ8 ($57.46 \to 21.60\text{ MiB}$), graph topology ($17.33\text{ MiB}$) rises from $23.2\%$ to **$44.5\%$** of the total structural footprint. This suggests diminishing returns from further vector-only compression because graph topology becomes an increasingly large fraction of total index memory.
> 4. **Motivation for Phase 6D (Off-Heap Graph Representation)**: Because graph topology becomes an increasingly large fraction of index memory (~44.5%), optimizing vector buffers alone yields diminishing returns. This motivates investigating off-heap graph representations (e.g. primitive flattened adjacency lists via `MemorySegment`) in Phase 6D.

### 4. Pareto Frontier Sweep & Two-Phase Search with FP32 Re-ranking

Evaluated using `ParetoFrontierRerankBenchmark` at $N = 10{,}000$, $D = 128$, $k = 10$, across beam widths $efSearch \in \{10, 20, 50, 100, 200, 400\}$:

> **Dominance Criterion**: A configuration dominates another on this benchmark if it achieves equal or higher Recall@10 at higher throughput (or lower latency), or higher throughput at equal or higher recall.

#### Pareto Frontier: Recall@10 vs Throughput (QPS) and Latency

| $efSearch$ | FP32 Recall | FP32 QPS | FP32 Mean Latency | SQ8 Recall | SQ8 QPS | SQ8 Mean Latency | Reranked Recall | Reranked QPS | Reranked Mean Latency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | 29.30% | 21,990.6 | 45.47 $\mu\text{s}$ | 29.77% | **30,654.1** | **32.62 $\mu\text{s}$** | 29.77% | 25,638.1 | 39.00 $\mu\text{s}$ |
| **20** | 44.77% | 12,430.3 | 80.45 $\mu\text{s}$ | 43.91% | **17,461.3** | **57.27 $\mu\text{s}$** | 43.91% | 14,546.0 | 68.75 $\mu\text{s}$ |
| **50** | 68.05% | 5,626.2 | 177.74 $\mu\text{s}$ | 67.34% | **8,035.8** | **124.44 $\mu\text{s}$** | 67.42% | 7,708.5 | 129.73 $\mu\text{s}$ |
| **100** | 85.47% | 3,042.2 | 328.71 $\mu\text{s}$ | 85.47% | **4,609.4** | **216.95 $\mu\text{s}$** | 85.70% | 3,204.6 | 312.05 $\mu\text{s}$ |
| **200** | 96.17% | 1,762.2 | 567.48 $\mu\text{s}$ | 95.23% | **2,688.4** | **371.97 $\mu\text{s}$** | **96.02%** | **2,271.3** | **440.28 $\mu\text{s}$** |
| **400** | 99.61% | 965.9 | 1,035.27 $\mu\text{s}$ | 98.36% | **1,361.0** | **734.74 $\mu\text{s}$** | **99.45%** | **1,108.9** | **901.80 $\mu\text{s}$** |

#### Two-Phase Re-ranking Breakdown: Candidate vs Final Recall

| $efSearch$ | Pure SQ8 Recall | Candidate Recall@10 | Re-ranked Recall | Recall Recovered | Unrecoverable Loss | Re-rank Overhead |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | 29.77% | 29.77% | 29.77% | 0.00% | 70.23% | 6.38 $\mu\text{s}$ |
| **20** | 43.91% | 43.91% | 43.91% | 0.00% | 56.09% | 11.48 $\mu\text{s}$ |
| **50** | 67.34% | 67.42% | 67.42% | +0.08% | 32.58% | 5.28 $\mu\text{s}$ |
| **100** | 85.47% | 85.70% | 85.70% | +0.23% | 14.30% | 95.11 $\mu\text{s}$ |
| **200** | 95.23% | 96.02% | **96.02%** | **+0.78%** | 3.98% | 68.31 $\mu\text{s}$ |
| **400** | 98.36% | 99.45% | **99.45%** | **+1.09%** | 0.55% | 167.06 $\mu\text{s}$ |

> [!TIP]
> **Mechanics of Two-Phase Search Recovery**:
> 1. **Observed Re-ranking Preservation**: In these benchmark runs, every ground-truth top-10 item recovered by the two-phase candidate set was retained by FP32 re-ranking ($\text{Candidate Recall@10} = \text{Re-ranked Recall@10}$ across all tested configurations); therefore, the observed re-ranking stage introduced no additional Recall@10 loss.
> 2. **Attribution of Remaining Recall Loss**: At $efSearch=400$, the candidate pool contained $99.45\%$ of true ground truth items. The remaining recall loss ($0.55\%$) is attributable to candidate generation/routing rather than final ranking in these benchmark configurations (items pruned during HNSW greedy routing that never entered the candidate beam).
> 3. **Trade-Off Profile**: At $efSearch=200$, Two-Phase Re-ranking achieves **comparable Recall@10 with higher throughput** ($96.02\%$ Recall vs FP32's $96.17\%$, lower by 0.15 percentage points in recall but $1.29\times$ faster: $2,271\text{ QPS}$ vs $1,762\text{ QPS}$). At $efSearch=400$, it reaches **$99.45\%$ Recall** at **$1,109\text{ QPS}$** ($15\%$ faster than FP32's $966\text{ QPS}$).

#### Latency Distribution: P50 / P95 / P99 ($\mu\text{s}$)

| $efSearch$ | FP32 HNSW (P50 / P95 / P99) | Pure SQ8 HNSW (P50 / P95 / P99) | SQ8 + FP32 Re-rank (P50 / P95 / P99) |
| :---: | :---: | :---: | :---: |
| **10** | 42.1 / 70.3 / 108.5 $\mu\text{s}$ | **28.7 / 60.2 / 86.1 $\mu\text{s}$** | 35.9 / 64.9 / 85.7 $\mu\text{s}$ |
| **20** | 76.4 / 108.0 / 136.4 $\mu\text{s}$ | **54.0 / 81.6 / 102.2 $\mu\text{s}$** | 65.1 / 95.8 / 115.9 $\mu\text{s}$ |
| **50** | 170.6 / 246.3 / 294.0 $\mu\text{s}$ | **117.6 / 174.0 / 226.8 $\mu\text{s}$** | 118.6 / 187.2 / 320.2 $\mu\text{s}$ |
| **100** | 295.4 / 557.3 / 675.9 $\mu\text{s}$ | **189.3 / 352.6 / 576.1 $\mu\text{s}$** | 288.3 / 493.5 / 693.3 $\mu\text{s}$ |
| **200** | 542.5 / 735.3 / 852.3 $\mu\text{s}$ | **347.7 / 514.4 / 708.1 $\mu\text{s}$** | 406.5 / 627.7 / 868.7 $\mu\text{s}$ |
| **400** | 979.7 / 1,364.1 / 1,695.0 $\mu\text{s}$ | **648.2 / 1,174.1 / 1,507.0 $\mu\text{s}$** | 810.8 / 1,334.1 / 1,676.7 $\mu\text{s}$ |

### 5. Systems Engineering Takeaways & Synthesis

1. **Quantization and Graph Navigability**:
   Phase 6C demonstrates experimentally that SQ8 quantization substantially reduces vector-memory consumption while preserving HNSW connectivity (0 isolated nodes, 1 component via BFS) and maintaining relatively small recall differences from the FP32 baseline across the tested scales. Quantization-induced topology divergence increases with scale (Layer-0 Jaccard similarity drops to $60.24\%$ at $100\text{K}$), yet the resulting recall gap remained below 1 percentage point ($\le 0.94\%$) in the tested configurations.
2. **Construction Throughput**:
   Pure SQ8 construction was $1.50\times - 1.87\times$ slower across tested scales. The additional cost is consistent with the current implementation's ADC distance calculations and neighbor-selection scratch dequantization.
3. **Topology Dominance (Remaining Memory Bottleneck After Vector Compression)**:
   At 100K vectors, SQ8 reduced measured JVM heap delta from 77.29 MiB to 42.01 MiB, while the measured structural footprint ($38.93\text{ MiB}$) closely matched the Phase 6B analytical model ($38.92\text{ MiB}$). Compressing vector storage shifts the primary memory component to the HNSW graph topology ($44.5\%$ of structural footprint), suggesting diminishing returns from further vector-only compression.
4. **Two-Phase Re-ranking Dynamics**:
   Two-phase FP32 re-ranking recovered most of the ranking loss observed with SQ8 candidates (recovering up to $+1.09\%$ at $efSearch=400$, reaching $99.45\%$ Recall@10 at $1,109\text{ QPS}$), indicating that the remaining recall loss in these experiments is primarily associated with candidate generation rather than final distance ordering.
5. **Workload Strategy Recommendations**:
   - *Throughput-Oriented Workloads*: Pure SQ8 HNSW delivers $1.39\times - 1.53\times$ higher QPS with $< 1\%$ recall difference from FP32.
   - *High-Recall Workloads ($\ge 99\%$)*: Two-Phase Re-ranking achieves $99.45\%$ Recall@10 while remaining $15\%$ faster than FP32 HNSW baseline.

---

## 🧱 Off-Heap / Foreign Function & Memory API Evaluation (Phase 6D)

Motivated by Phase 6C's discovery of **Graph Topology Dominance** (where the HNSW graph topology consumed 44.5% of total index memory at $100\text{K}$, leaving hundreds of thousands of heap objects in the JVM heap), Phase 6D evaluates the standard **Java 25 Foreign Function & Memory (FFM) API (`java.lang.foreign.MemorySegment`)** for off-heap vector indexing (`OffHeapQuantizedHnswIndex`), investigating four empirical systems questions:

> 1. **Heap Delta Reduction**: To what extent does moving graph topology and quantized vectors off-heap reduce JVM heap residency?
> 2. **Memory Accounting**: Does native memory allocation match structural expectations without hidden physical RAM bloat?
> 3. **FFM Access Penalty**: What is the query throughput and latency cost of FFM bounds checking and pointer-free arithmetic?
> 4. **Behavioral Parity**: Does off-heap routing produce the same edges and candidate decisions as the on-heap reference index?

### 1. Three-Way Memory Footprint Characterization ($D=128$)

| Scale ($N$) | FP32 Heap Delta | On-Heap SQ8 Delta | Off-Heap Heap Delta | Native Allocated | Measured Process RSS | Heap Delta Red vs SQ8 | Heap Delta Red vs FP32 | Native Accounting |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1,000** | 0.93 MiB | 0.45 MiB | **0.15 MiB** | 0.36 MiB | 0.52 MiB | **2.94×** | **6.03×** | Verified (`true`) |
| **10,000** | 10.62 MiB | 3.92 MiB | **0.78 MiB** | 3.57 MiB | 4.36 MiB | **5.00×** | **13.55×** | Verified (`true`) |
| **50,000** | 40.88 MiB | 21.26 MiB | **4.21 MiB** | 17.84 MiB | 22.05 MiB | **5.04×** | **9.70×** | Verified (`true`) |
| **100,000** | 77.29 MiB | 42.02 MiB | **8.24 MiB** | 35.67 MiB | 43.92 MiB | **5.10×** | **9.37×** | Verified (`true`) |

- **Measured JVM Heap Delta giảm 5.10× ở $N=100\text{K}$**: Từ $42.02\text{ MiB}$ với On-Heap SQ8 xuống **$8.24\text{ MiB}$** với Off-Heap SQ8 ($9.37\times$ so với FP32 HNSW ở $77.29\text{ MiB}$). Đây là empirical runtime observation thông qua GC delta methodology, xác nhận phần lớn vector và graph payload đã được chuyển ra ngoài heap do JVM quản lý.
- **Measured Off-Heap Native Allocation + Process Memory Observation**: Tổng quan sát RSS process là $43.92\text{ MiB}$ (so với $42.02\text{ MiB}$ On-Heap SQ8). Sự chênh lệch này phản ánh việc cấp phát bộ nhớ native cùng các đặc tính căn chỉnh bộ nhớ và allocator layout.

### 2. Search Latency Distribution, Throughput & Parity ($N=10{,}000, D=128, k=10$)

| $efSearch$ | FP32 QPS | On-Heap SQ8 QPS | Off-Heap SQ8 QPS | Throughput Ratio (Off/On) | Off-Heap Re-ranked QPS | Top-10 Parity Agreement (On vs Off) | Pure SQ8 Recall@10 (vs Oracle) | Re-ranked Recall@10 (vs Oracle) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **10** | 20,865.9 | 27,805.1 | 24,057.5 | **0.87×** | 20,174.3 | **93.28%** | 29.45% | 29.45% |
| **20** | 11,136.1 | 15,986.0 | 15,382.4 | **0.96×** | 13,900.0 | **93.91%** | 44.30% | 44.30% |
| **50** | 5,576.3 | 7,297.9 | 7,287.7 | **1.00×** | 6,166.5 | **96.56%** | 67.11% | 67.19% |
| **100** | 3,264.6 | 4,158.0 | 4,025.6 | **0.97×** | 3,064.5 | **99.22%** | 85.31% | 85.55% |
| **200** | 1,673.3 | 2,271.9 | 2,110.4 | **0.93×** | 1,790.9 | **99.69%** | 95.39% | 96.17% |
| **400** | 999.8 | 1,312.6 | 1,222.7 | **0.93×** | 962.4 | **99.92%** | 98.44% | 99.53% |

- **Search Throughput Ratio (0.87–1.00×)**: Tốc độ truy vấn của Off-Heap SQ8 dao động trong khoảng 0.87× đến 1.00× so với On-Heap SQ8 baseline. Ở cấu hình tiêu chuẩn $efSearch=50$, thông lượng thực tế gần như tương đương (7,287.7 QPS vs 7,297.9 QPS; P50: $131.9\ \mu\text{s}$ vs $132.5\ \mu\text{s}$).
- **Behavioral Parity ≠ Ground Truth Recall**:
  - *Behavioral Parity (On-Heap SQ8 vs Off-Heap SQ8)*: Đo lường mức độ đồng thuận lựa chọn ứng viên giữa 2 biểu diễn bộ nhớ, đạt **$96.56\%$ ở $efSearch=50$** và tăng lên **$99.92\%$ ở $efSearch=400$**.
  - *Recall@10 (vs FP32 Flat Oracle)*: Đo lường độ chính xác so với Ground Truth thực tế ($98.44\%$ với pure SQ8 và phục hồi lên **$99.53\%$ với Two-Phase Re-ranking**, so với FP32 baseline $99.61\%$).
- **Core Systems Takeaway**:
  > **Off-heap storage substantially reduced JVM heap residency, but did not produce a corresponding search-throughput improvement under the tested workload. The primary demonstrated benefit is memory-domain separation rather than raw query acceleration.**

### 3. Construction Throughput & GC Profile

- **Index Construction**: Ở quy mô $10{,}000$ vectors, thời gian xây dựng Off-Heap SQ8 ($14,059\text{ ms}$, $711.3\text{ vec/s}$) chỉ chậm hơn On-Heap SQ8 khoảng **$1.06\times$** ($13,222\text{ ms}$, $756.3\text{ vec/s}$), cho thấy chi phí đóng gói layout native tương đối nhỏ so với tổng chi phí tính toán lượng tử và heuristic HNSW.
- **JMH GC Allocation Profile (`-prof gc`)**:
  - *Distance Kernel*: Không ghi nhận allocation đo lường được ở độ phân giải benchmark (báo cáo $0.000\text{ B/op}$).
  - *Search Traversal*: Vẫn tạo một số temporary objects trên JVM heap trong quá trình duyệt đồ thị; trong cấu hình benchmark này, chúng không được quan sát là gây Old-Gen promotion.
  - *GC Impact*: Việc chuyển phần vector/graph payload lớn ra off-heap giúp **giảm đáng kể lượng dữ liệu lâu dài mà GC-managed heap phải quản lý**.

---

## 🚀 Getting Started

### Prerequisites
- JDK 25 or higher (compiled with `--release 25`).
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

This runs all **333 unit and integration tests** (Core: 238, Persistence: 58, Benchmark: 37; 0 failures, 0 errors, 0 skipped) across Linux and Windows CI.

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

# Phase 6A: Profile memory footprint (analytical vs empirical GC delta) across scales
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar com.nanovector.benchmark.memory.MemoryFootprintProfiler

# Phase 6A: Profile query throughput and latency distribution (P50/P90/P99) across scales
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar com.nanovector.benchmark.scale.ScaleLatencySearchBenchmark --latency-profile

# Phase 6A: Profile HNSW construction throughput and multi-layer topology properties
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar com.nanovector.benchmark.scale.ScaleConstructionBenchmark --topology-profile

# Phase 6B: Measure quantization-induced Recall@10 trade-off across scales 1K to 100K
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar com.nanovector.benchmark.quantization.Sq8RecallAccuracyBenchmark

# Phase 6B: Benchmark Flat vs QuantizedFlat throughput and latency distribution across scales
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar com.nanovector.benchmark.quantization.FlatVsQuantizedFlatBenchmark

# Phase 6C: Benchmark HNSW construction strategies, topology divergence, and recall attribution across scales
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar com.nanovector.benchmark.topology.HnswConstructionStrategyBenchmark

# Phase 6C: Profile Quantized HNSW memory footprint (structural vs empirical heap delta) across scales
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar com.nanovector.benchmark.memory.MemoryFootprintProfiler --quantized-hnsw

# Phase 6C: Sweep Pareto frontier and evaluate Two-Phase search with FP32 re-ranking
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar com.nanovector.benchmark.search.ParetoFrontierRerankBenchmark

# Phase 6D: Profile 3-Way Memory Footprint (FP32 vs On-Heap SQ8 vs Off-Heap SQ8) across scales 1K to 100K
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar com.nanovector.benchmark.memory.MemoryFootprintProfiler --phase6d

# Phase 6D: Benchmark On-Heap vs Off-Heap SQ8 search latency, throughput, parity, and construction cost
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar com.nanovector.benchmark.search.OnHeapVsOffHeapSearchBenchmark 10000

# Phase 6D: Profile Off-Heap vs On-Heap GC allocation rates and churn with JMH GC profiler
java --add-modules jdk.incubator.vector -cp nanovector-benchmark/target/benchmarks.jar org.openjdk.jmh.Main com.nanovector.benchmark.allocation.OffHeapAllocationBenchmark -prof gc
```

---

## 💻 Standalone Command-Line Interface (`nanovector-cli`)

`nanovector-cli` provides an allocation-conscious, scriptable command-line tool powered by Picocli for operating on `.nvec` binary index files without running a web server.

### Building and Running the CLI
```bash
./mvnw package -pl nanovector-cli -am -DskipTests
java --add-modules jdk.incubator.vector -jar nanovector-cli/target/nanovector-cli-0.1.0-SNAPSHOT.jar [command] [options]
```

### Supported CLI Commands:
1. **`create`**: Initialize and persist an empty `.nvec` index file.
   ```bash
   # Create an HNSW index with Euclidean metric and dimension 128
   java --add-modules jdk.incubator.vector -jar nanovector-cli.jar create -d 128 -m EUCLIDEAN -t HNSW index.nvec
   ```
2. **`inspect`**: Verify file integrity via hardware CRC32C and display format metadata.
   ```bash
   java --add-modules jdk.incubator.vector -jar nanovector-cli.jar inspect index.nvec
   ```
3. **`insert`**: Ingest vectors from a CSV file into an existing `.nvec` index.
   ```bash
   java --add-modules jdk.incubator.vector -jar nanovector-cli.jar insert index.nvec --csv vectors.csv
   ```
4. **`query`**: Execute k-NN similarity search against an `.nvec` index.
   ```bash
   java --add-modules jdk.incubator.vector -jar nanovector-cli.jar query index.nvec -k 5 --vector "0.1,0.2,...,0.5"
   ```

---

## 🌐 High-Performance REST API Microservice (`nanovector-server`)

`nanovector-server` exposes NanoVector's core vector search engine as a concurrent HTTP microservice built on Spring Boot 3.4.0 and Java 25 LTS.

### Key Architectural Characteristics
1. **Thread-Safe Search Traversal & Context Pooling**:
   - Concurrent queries obtain independent `EpochVisitedSet` instances from a non-blocking object pool (`SearchContextProvider`) residing directly in `nanovector-core`.
   - Index instances are natively thread-safe without requiring Spring or external synchronizers.
2. **Fair Read-Write Locking & Native Lifecycle Draining (`DefaultManagedIndex`)**:
   - Synchronizes concurrent readers and mutating batch writers using fair `ReentrantReadWriteLock(true)`.
   - Idempotent `close()` sets an atomic flag rejecting new requests immediately and acquires the exclusive write lock to **drain all in-flight reader threads before closing underlying native FFM arenas or file descriptors**, completely eliminating JVM use-after-free crashes.
   - **Save** operations hold the read lock throughout the entire duration of binary serialization (`NvecWriter`), guaranteeing consistent disk snapshots without torn state.
3. **Storage Security & Path Traversal Prevention**:
   - Clients interact with index names or identifiers; arbitrary filesystem paths are strictly disallowed.
   - Files are resolved inside the designated server storage directory (`nanovector.server.data-dir`), verifying `resolved.startsWith(baseDir)` and stripping path traversal tokens (`..`, `/`, `\`).
4. **Interactive OpenAPI 3.0 & Swagger UI**:
   - **Dashboard**: `http://localhost:8080/swagger-ui.html`
   - **OpenAPI Schema**: `http://localhost:8080/v3/api-docs`
5. **Operational Health Indicator**:
   - Spring Boot Actuator endpoint `GET /actuator/health` reports service readiness and registered index counts without leaking sensitive internal paths or memory layout specifics.

### REST API Endpoints Overview

| Method | Endpoint | HTTP Status | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/indexes` | `201 Created` | Create and register a new vector index (`FLAT`, `HNSW`, `HNSW_SQ8`, `HNSW_SQ8_OFFHEAP`) |
| `GET` | `/api/v1/indexes` | `200 OK` | List metadata of all registered indexes |
| `GET` | `/api/v1/indexes/{name}` | `200 OK` | Get index metadata descriptor |
| `DELETE` | `/api/v1/indexes/{name}` | `204 No Content` | Delete and drain index, deallocating native resources |
| `POST` | `/api/v1/indexes/{name}/vectors` | `200 OK` | Batch insert vectors with atomic pre-validation |
| `POST` | `/api/v1/indexes/{name}/query` | `200 OK` | Execute thread-safe k-NN query (optional `efSearch` override) |
| `POST` | `/api/v1/indexes/{name}/save` | `200 OK` | Persist index to `.nvec` format in server data directory |
| `POST` | `/api/v1/indexes/load` | `201 Created` | Restore index from `.nvec` file into active registry |

> [!IMPORTANT]
> **Compilation & Runtime Compatibility Note on Java 25 & Spring Boot 3.4.0**:
> `nanovector-core`, `nanovector-persistence`, `nanovector-benchmark`, and `nanovector-cli` are compiled targeting `--release 25`.
> `nanovector-server` targets `--release 24` (`<maven.compiler.release>24</maven.compiler.release>`) to maintain binary compatibility with Spring Boot 3.4.0's embedded ASM `ClassReader` (which validates major version $\le 68$), while running on **Java 25 LTS Runtime** with incubator SIMD Vector API (`--add-modules jdk.incubator.vector`) and Foreign Function & Memory (FFM) API.
> When Spring Framework upgrades its embedded ASM parser to support major version 69, `nanovector-server` can be bumped to `--release 25`.
> Zero Spring or OpenAPI dependencies leak into the core, persistence, benchmark, or CLI modules.

*Detailed deployment instructions, curl examples, and configuration options are available in [OPERATIONS_GUIDE.md](file:///d:/NanoVector/NanoVector/nanovector-server/OPERATIONS_GUIDE.md).*

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
- [x] **v0.6 (Phase 6A)**: Scale & Memory Characterization ($N \in \{1\text{K}, 10\text{K}, 50\text{K}, 100\text{K}\}$ at $D=128$, 17 benchmark tests, 221 total):
  - [x] Analytical structural model vs empirical heap delta profiling (`MemoryFootprintProfiler`).
  - [x] Flat vs HNSW query latency distribution (P50/P90/P99) and throughput scaling (`ScaleLatencySearchBenchmark`).
  - [x] HNSW graph construction scaling and multi-layer topology characterization (`ScaleConstructionBenchmark`).
- [x] **v0.6 (Phase 6B)**: Quantization study (FP32 vs SQ8 Scalar Quantization, memory reduction vs recall trade-off, 24 benchmark tests, 240 total):
  - [x] Asymmetric per-vector affine SQ8 quantizer with mathematical verification (`AsymmetricSq8Quantizer`).
  - [x] Contiguous primitive storage with zero-allocation access (`QuantizedVectorStorage`).
  - [x] Asymmetric Distance Computation (ADC) engine with Java Vector API SIMD acceleration (`VectorQuantizedEuclideanDistance`).
  - [x] Ground truth recall evaluation proving $\le 1.09\%$ quantization loss and 100% scalar-SIMD parity (`Sq8RecallAccuracyBenchmark`).
  - [x] Memory footprint characterization verifying $3.76\times$ payload reduction and $2.36\times - 2.44\times$ measured heap delta reduction.
  - [x] Throughput and latency distribution benchmarking characterizing CPU-bound scalar widening and SIMD memory-traffic bottleneck transition.
- [x] **v0.6 (Phase 6C)**: Quantized HNSW Systems Study (HNSW + SQ8 integration, topology divergence, Pareto frontier, 30 benchmark tests, 263 total):
  - [x] Decoupled `QuantizedHnswIndex` implementation with SIMD ADC distance hot path and thread-safe scratch dequantization.
  - [x] Graph construction strategy benchmark (`HnswConstructionStrategyBenchmark`) evaluating build time ($1.50\times - 1.87\times$), Layer-0 Jaccard divergence ($96.07\% \to 60.24\%$), and 100% invariant preservation (1 component, 0 isolated nodes).
  - [x] Mathematical recall loss attribution proving $\le 0.94\%$ total recall loss vs FP32 HNSW across scales $1\text{K} \to 100\text{K}$.
  - [x] Measured scale memory characterization validating Phase 6B structural model ($38.93\text{ MiB}$ at 100K) and quantifying Graph Topology Dominance ($44.5\%$ of index footprint).
- [x] **v0.6 (Phase 6D)**: Off-Heap / Foreign Function & Memory API (`MemorySegment`) Systems Study (37 benchmark tests, 333 total):
  - [x] Baseline upgrade to **Java 25 LTS** (`maven.compiler.release=25`) and JaCoCo `0.8.14`.
  - [x] Contiguous interleaved native vector storage (`OffHeapQuantizedVectorStorage`, $136\text{ B/vec}$ at $D=128$).
  - [x] Pointer-free flattened multi-layer graph topology (`OffHeapGraphLayout` and `OffHeapHnswGraph`).
  - [x] Native SIMD ADC Euclidean distance kernel (`OffHeapQuantizedEuclideanDistance` with `ByteVector.fromMemorySegment`).
  - [x] End-to-end off-heap HNSW index (`OffHeapQuantizedHnswIndex`) supporting single-phase search and two-phase re-ranking.
  - [x] Behavioral parity verification proving identical edge-for-edge and candidate-for-candidate routing equivalence.
  - [x] Three-way memory footprint characterization across scales $1\text{K} \to 100\text{K}$ measuring a **$5.10\times$ reduction in measured JVM heap delta** ($42.02\text{ MiB} \to 8.24\text{ MiB}$).
  - [x] Empirical search latency and throughput benchmark characterizing **$0.87\times - 1.00\times$ throughput ratio vs on-heap SQ8**, with $93.28\% - 99.92\%$ behavioral top-10 parity agreement.
  - [x] JMH GC allocation profiling showing no measurable allocation at JMH resolution ($0.000\text{ B/op}$ reported) on distance kernel and reducing heap data under GC management.
- [x] **v0.7 (Phase 7)**: Standalone CLI, Spring Boot REST API & Systems Deployment (410 unit/integration tests total):
  - [x] **Phase 7A (CLI)**: Standalone Picocli command-line interface (`nanovector-cli`) with `create`, `inspect`, `insert`, and `query` commands and real-process E2E testing.
  - [x] **Phase 7B (Server Foundation)**: Central `IndexRegistry`, `ManagedIndex` fair read-write locking, `SearchContextProvider` object pool for thread-safe search contexts, and 20-thread concurrency verification on both on-heap and off-heap indexes with 100% ground-truth parity.
  - [x] **Phase 7C (REST API)**: Full REST API under `/api/v1/indexes`, atomic batch pre-validation, k-NN queries with `efSearch` override, secure `.nvec` persistence without arbitrary filesystem access, and `@RestControllerAdvice` error response normalization.
  - [x] **Phase 7D (OpenAPI & Operations)**: Springdoc OpenAPI 3.0 / Swagger UI (`/swagger-ui.html`), Actuator health check (`/actuator/health`), comprehensive operational documentation (`OPERATIONS_GUIDE.md`), and Java 25 / release 24 binary compatibility verification.





