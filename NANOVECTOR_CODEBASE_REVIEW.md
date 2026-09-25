# 🚀 NANOVECTOR: BÁO CÁO TOÀN DIỆN MÃ NGUỒN & TÀI LIỆU REVIEW KIẾN TRÚC

Tài liệu này được tạo ra nhằm cung cấp cái nhìn 360 độ về dự án **NanoVector**: toàn bộ nội dung tài liệu thiết kế (`README.md`), cây thư mục dự án, và mã nguồn đầy đủ của tất cả các file cốt lõi (Core Engine, Storage, Distance Calculators, Graph Traversal, Index Orchestration).

---

# MỤC LỤC
1. [Nội Dung README.md Chính Thức](#1-nội-dung-readmemd-chính-thức)
2. [Cây Thư Mục & Cấu Trúc File Dự Án](#2-cây-thư-mục--cấu-trúc-file-dự-án)
3. [Mã Nguồn Các File Cốt Lõi (Core Engine)](#3-mã-nguồn-các-file-cốt-lõi-core-engine)
   - [3.1. Hợp đồng Interface & Data Model](#31-hợp-đồng-interface--data-model)
     - `VectorIndex.java`
     - `SearchResult.java`
   - [3.2. Quản lý Bộ nhớ & Lưu trữ Nguyên thủy](#32-quản-lý-bộ-nhớ--lưu-trữ-nguyên-thủy)
     - `VectorStorage.java`
     - `VectorUtils.java`
   - [3.3. Tầng Tính Toán Khoảng Cách (Scalar & SIMD Hardware Acceleration)](#33-tầng-tính-toán-khoảng-cách-scalar--simd-hardware-acceleration)
     - `DistanceMetric.java`
     - `DistanceCalculator.java`
     - `ScalarEuclideanDistance.java`
     - `VectorEuclideanDistance.java`
     - `VectorCosineDistance.java`
     - `VectorDotProductDistance.java`
   - [3.4. Chỉ Mục Tham Chiếu (Ground Truth Oracle)](#34-chỉ-mục-tham-chiếu-ground-truth-oracle)
     - `FlatIndex.java`
   - [3.5. Cấu Trúc Dữ Liệu & Giải Thuật HNSW](#35-cấu-trúc-dữ-liệu--giải-thuật-hnsw)
     - `HnswConfig.java`
     - `HnswNode.java`
     - `BoundedMaxHeap.java`
     - `EpochVisitedSet.java`
     - `LevelGenerator.java`
     - `NeighborSelector.java`
     - `HnswGraph.java`
   - [3.6. Bộ Điều Phối Chỉ Mục HNSW](#36-bộ-điều-phối-chỉ-mục-hnsw)
     - `HnswIndex.java`
4. [Tài Liệu Hướng Dẫn Review Chuyên Sâu](#4-tài-liệu-hướng-dẫn-review-chuyên-sâu)

---

# 1. NỘI DUNG README.md CHÍNH THỨC

```markdown
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
- [ ] **v0.4 (Phase 4)**: Binary persistence (`.nvec` file format).
- [ ] **v0.5 (Phase 5)**: Rigorous benchmark suite (JMH microbenchmarking, Scalar vs SIMD topology/recall comparison, scale profiling).
- [ ] **v0.6 (Phase 6)**: Scale & memory experiments (quantization study, off-heap MemorySegment evaluation).
- [ ] **v0.7 (Phase 7)**: Standalone CLI & Spring Boot REST API.
```

---

# 2. CÂY THƯ MỤC & CẤU TRÚC FILE DỰ ÁN

```text
D:\NanoVector\NanoVector
│   .gitignore
│   LICENSE
│   mvnw
│   mvnw.cmd
│   pom.xml
│   README.md
│
├───.github
│   └───workflows
│           ci.yml
│
├───.githooks
│       pre-commit
│       pre-push
│
└───nanovector-core
    │   pom.xml
    │
    └───src
        ├───main
        │   └───java
        │       └───com
        │           └───nanovector
        │               └───core
        │                   ├───distance
        │                   │       DistanceCalculator.java
        │                   │       DistanceMetric.java
        │                   │       ScalarCosineDistance.java
        │                   │       ScalarDotProductDistance.java
        │                   │       ScalarEuclideanDistance.java
        │                   │       VectorCosineDistance.java
        │                   │       VectorDotProductDistance.java
        │                   │       VectorEuclideanDistance.java
        │                   │
        │                   ├───heap
        │                   │       BoundedMaxHeap.java
        │                   │
        │                   ├───hnsw
        │                   │       EpochVisitedSet.java
        │                   │       HnswConfig.java
        │                   │       HnswGraph.java
        │                   │       HnswNode.java
        │                   │       LevelGenerator.java
        │                   │       NeighborSelector.java
        │                   │
        │                   ├───index
        │                   │       FlatIndex.java
        │                   │       HnswIndex.java
        │                   │       VectorIndex.java
        │                   │
        │                   ├───model
        │                   │       SearchResult.java
        │                   │
        │                   ├───storage
        │                   │       VectorStorage.java
        │                   │
        │                   └───util
        │                           VectorUtils.java
        │
        └───test
            └───java
                └───com
                    └───nanovector
                        └───core
                            ├───benchmark
                            │       HnswBenchmark.java
                            │       SimdBenchmark.java
                            │
                            ├───distance
                            │       DistanceCalculatorTest.java
                            │       SimdScalarEquivalenceTest.java
                            │
                            ├───heap
                            │       BoundedMaxHeapTest.java
                            │
                            ├───hnsw
                            │       EpochVisitedSetTest.java
                            │       HnswGraphInvariantTest.java
                            │       HnswGraphTest.java
                            │       LevelGeneratorTest.java
                            │       NeighborSelectorTest.java
                            │       SearchLayerTest.java
                            │
                            ├───index
                            │       FlatIndexTest.java
                            │       HnswIndexTest.java
                            │       HnswRecallTest.java
                            │
                            ├───storage
                            │       VectorStorageTest.java
                            │
                            ├───util
                            │       VectorUtilsTest.java
                            │
                            └───vector
                                    VectorApiAvailabilityTest.java
```

---

# 3. MÃ NGUỒN CÁC FILE CỐT LÕI (CORE ENGINE)

---

## 3.1. Hợp đồng Interface & Data Model

### `VectorIndex.java`
```java
package com.nanovector.core.index;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.model.SearchResult;
import java.util.List;

/** Common contract for vector similarity search indexes. */
public interface VectorIndex {

  /**
   * Inserts a vector with a unique external identifier into the index.
   *
   * @param id unique external ID
   * @param vector float array vector matching the index dimension
   * @throws IllegalArgumentException if vector length mismatch, contains NaN/Inf, or ID exists
   */
  void insert(long id, float[] vector);

  /**
   * Finds the Top-K nearest neighbors to the query vector.
   *
   * @param query query vector matching the index dimension
   * @param k number of nearest neighbors to return (must be > 0)
   * @return sorted list of {@link SearchResult} ordered by distance ascending
   */
  List<SearchResult> searchKnn(float[] query, int k);

  /** Returns the total number of indexed vectors. */
  int size();

  /** Returns the dimensionality of vectors in this index. */
  int dimension();

  /** Returns the distance metric used by this index. */
  DistanceMetric metric();
}
```

### `SearchResult.java`
```java
package com.nanovector.core.model;

import java.util.Objects;

/**
 * Immutable search result representing a retrieved vector and its distance to the query.
 *
 * <p>Implements {@link Comparable} based on the <b>Distance Minimization</b> contract: results with
 * smaller distances rank higher (come first).
 *
 * <p><b>Deterministic tie-breaking:</b> When distances are identical, ranking defaults to {@code
 * id} ascending, ensuring 100% reproducible ordering.
 *
 * @param id the unique external identifier of the retrieved vector
 * @param distance the calculated distance from the query vector (smaller = more similar)
 */
public record SearchResult(long id, float distance) implements Comparable<SearchResult> {

  @Override
  public int compareTo(SearchResult other) {
    Objects.requireNonNull(other, "Cannot compare SearchResult with null");
    int cmp = Float.compare(this.distance, other.distance);
    if (cmp != 0) {
      return cmp;
    }
    return Long.compare(this.id, other.id);
  }
}
```

---

## 3.2. Quản lý Bộ nhớ & Lưu trữ Nguyên thủy

### `VectorStorage.java`
```java
package com.nanovector.core.storage;

import com.nanovector.core.util.VectorUtils;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * Contiguous primitive-based vector storage.
 *
 * <p>Maintains vectors in a single continuous 1D {@code float[]} array:
 *
 * <pre>
 *   offset = internalId * dimension
 * </pre>
 *
 * Manages two-way mapping between external {@code long} IDs and internal {@code int} indices.
 * Rejects duplicate external IDs in Phase 1 with {@link IllegalArgumentException}.
 */
public final class VectorStorage {

  private static final int DEFAULT_INITIAL_CAPACITY = 1024;

  private final int dimension;
  private float[] vectors;
  private long[] externalIds;
  private final Map<Long, Integer> externalToInternal;
  private int size;

  public VectorStorage(int dimension) {
    this(dimension, DEFAULT_INITIAL_CAPACITY);
  }

  public VectorStorage(int dimension, int initialCapacity) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (initialCapacity <= 0) {
      throw new IllegalArgumentException("Initial capacity must be positive: " + initialCapacity);
    }
    this.dimension = dimension;
    this.vectors = new float[initialCapacity * dimension];
    this.externalIds = new long[initialCapacity];
    this.externalToInternal = HashMap.newHashMap(initialCapacity);
    this.size = 0;
  }

  public synchronized int insert(long externalId, float[] vector) {
    VectorUtils.checkDimension(vector, dimension);
    VectorUtils.checkFinite(vector);

    if (externalToInternal.containsKey(externalId)) {
      throw new IllegalArgumentException("Duplicate external ID: " + externalId);
    }

    ensureCapacity(size + 1);

    int internalId = size;
    int offset = internalId * dimension;
    System.arraycopy(vector, 0, vectors, offset, dimension);
    externalIds[internalId] = externalId;
    externalToInternal.put(externalId, internalId);

    size++;
    return internalId;
  }

  public long getExternalId(int internalId) {
    checkInternalBounds(internalId);
    return externalIds[internalId];
  }

  public int getInternalId(long externalId) {
    Integer internalId = externalToInternal.get(externalId);
    if (internalId == null) {
      throw new NoSuchElementException("External ID not found: " + externalId);
    }
    return internalId;
  }

  public boolean contains(long externalId) {
    return externalToInternal.containsKey(externalId);
  }

  public float[] getVector(int internalId) {
    checkInternalBounds(internalId);
    float[] copy = new float[dimension];
    System.arraycopy(vectors, internalId * dimension, copy, 0, dimension);
    return copy;
  }

  public void copyVector(int internalId, float[] dest) {
    checkInternalBounds(internalId);
    VectorUtils.checkDimension(dest, dimension);
    System.arraycopy(vectors, internalId * dimension, dest, 0, dimension);
  }

  public float[] getVectorBuffer() {
    return vectors;
  }

  public int getOffset(int internalId) {
    checkInternalBounds(internalId);
    return internalId * dimension;
  }

  public int size() {
    return size;
  }

  public int dimension() {
    return dimension;
  }

  public int capacity() {
    return externalIds.length;
  }

  private void ensureCapacity(int minCapacity) {
    int currentCapacity = externalIds.length;
    if (minCapacity > currentCapacity) {
      int newCapacity = Math.max(minCapacity, currentCapacity * 2);
      vectors = Arrays.copyOf(vectors, newCapacity * dimension);
      externalIds = Arrays.copyOf(externalIds, newCapacity);
    }
  }

  private void checkInternalBounds(int internalId) {
    if (internalId < 0 || internalId >= size) {
      throw new IndexOutOfBoundsException(
          "Internal ID out of bounds: " + internalId + ", current size: " + size);
    }
  }
}
```

### `VectorUtils.java`
```java
package com.nanovector.core.util;

import java.util.Objects;

/** Utility methods for vector operations and validation. */
public final class VectorUtils {

  public static final float ZERO_NORM_THRESHOLD = 1e-9f;

  private static final String VECTOR_NOT_NULL_MSG = "Vector must not be null";

  private VectorUtils() {}

  public static void checkDimension(float[] vector, int expectedDimension) {
    Objects.requireNonNull(vector, VECTOR_NOT_NULL_MSG);
    if (vector.length != expectedDimension) {
      throw new IllegalArgumentException(
          "Invalid vector dimension: expected " + expectedDimension + ", got " + vector.length);
    }
  }

  public static void checkFinite(float[] vector) {
    Objects.requireNonNull(vector, VECTOR_NOT_NULL_MSG);
    for (int i = 0; i < vector.length; i++) {
      if (!Float.isFinite(vector[i])) {
        throw new IllegalArgumentException(
            "Vector component at index " + i + " is non-finite: " + vector[i]);
      }
    }
  }

  public static float squaredNorm(float[] vector) {
    Objects.requireNonNull(vector, VECTOR_NOT_NULL_MSG);
    checkFinite(vector);

    double sumSq = 0.0;
    for (float v : vector) {
      sumSq += (double) v * v;
    }

    if (sumSq == 0.0) {
      throw new IllegalArgumentException("Cannot normalize zero-vector");
    }

    float norm = (float) Math.sqrt(sumSq);
    float[] normalized = new float[vector.length];
    for (int i = 0; i < vector.length; i++) {
      normalized[i] = vector[i] / norm;
    }
    return normalized;
  }
}
```

---

## 3.3. Tầng Tính Toán Khoảng Cách (Scalar & SIMD Hardware Acceleration)

### `DistanceMetric.java`
```java
package com.nanovector.core.distance;

public enum DistanceMetric {
  EUCLIDEAN,
  COSINE,
  DOT_PRODUCT
}
```

### `DistanceCalculator.java`
```java
package com.nanovector.core.distance;

public interface DistanceCalculator {

  float distance(float[] a, float[] b);

  float distance(float[] buffer, int offset, float[] query);

  float distance(float[] bufferA, int offsetA, float[] bufferB, int offsetB, int length);

  default float distance(float[] buffer, int offsetA, int offsetB, int length) {
    return distance(buffer, offsetA, buffer, offsetB, length);
  }

  DistanceMetric metric();

  static DistanceCalculator create(DistanceMetric metric, boolean useSimd) {
    java.util.Objects.requireNonNull(metric, "Metric must not be null");
    if (useSimd) {
      return switch (metric) {
        case EUCLIDEAN -> new VectorEuclideanDistance();
        case COSINE -> new VectorCosineDistance();
        case DOT_PRODUCT -> new VectorDotProductDistance();
      };
    } else {
      return switch (metric) {
        case EUCLIDEAN -> new ScalarEuclideanDistance();
        case COSINE -> new ScalarCosineDistance();
        case DOT_PRODUCT -> new ScalarDotProductDistance();
      };
    }
  }

  static DistanceCalculator create(DistanceMetric metric) {
    return create(metric, true);
  }
}
```

### `VectorEuclideanDistance.java` (SIMD)
```java
package com.nanovector.core.distance;

import java.util.Objects;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class VectorEuclideanDistance implements DistanceCalculator {

  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  @Override
  public float distance(float[] a, float[] b) {
    Objects.requireNonNull(a, "Vector 'a' must not be null");
    Objects.requireNonNull(b, "Vector 'b' must not be null");
    if (a.length != b.length) {
      throw new IllegalArgumentException(
          "Vector dimension mismatch: a.length=" + a.length + ", b.length=" + b.length);
    }
    return distance(a, 0, b, 0, a.length);
  }

  @Override
  public float distance(float[] buffer, int offset, float[] query) {
    Objects.requireNonNull(query, "Query vector must not be null");
    return distance(buffer, offset, query, 0, query.length);
  }

  @Override
  public float distance(float[] bufferA, int offsetA, float[] bufferB, int offsetB, int length) {
    Objects.requireNonNull(bufferA, "Buffer A must not be null");
    Objects.requireNonNull(bufferB, "Buffer B must not be null");
    if (length < 0) {
      throw new IllegalArgumentException("Length must be non-negative: " + length);
    }
    if (offsetA < 0 || offsetA + length > bufferA.length) {
      throw new IndexOutOfBoundsException("OffsetA and length exceed bufferA bounds");
    }
    if (offsetB < 0 || offsetB + length > bufferB.length) {
      throw new IndexOutOfBoundsException("OffsetB and length exceed bufferB bounds");
    }

    int upperBound = SPECIES.loopBound(length);
    FloatVector sumVec = FloatVector.zero(SPECIES);

    int i = 0;
    for (; i < upperBound; i += SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(SPECIES, bufferA, offsetA + i);
      FloatVector vb = FloatVector.fromArray(SPECIES, bufferB, offsetB + i);
      FloatVector diff = va.sub(vb);
      sumVec = diff.fma(diff, sumVec);
    }

    float sum = sumVec.reduceLanes(VectorOperators.ADD);

    // Scalar tail loop
    for (; i < length; i++) {
      float diff = bufferA[offsetA + i] - bufferB[offsetB + i];
      sum += diff * diff;
    }

    return sum;
  }

  @Override
  public DistanceMetric metric() {
    return DistanceMetric.EUCLIDEAN;
  }
}
```

### `VectorCosineDistance.java` (SIMD)
```java
package com.nanovector.core.distance;

import java.util.Objects;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class VectorCosineDistance implements DistanceCalculator {

  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  @Override
  public float distance(float[] a, float[] b) {
    Objects.requireNonNull(a, "Vector 'a' must not be null");
    Objects.requireNonNull(b, "Vector 'b' must not be null");
    if (a.length != b.length) {
      throw new IllegalArgumentException("Vector dimension mismatch");
    }
    return distance(a, 0, b, 0, a.length);
  }

  @Override
  public float distance(float[] buffer, int offset, float[] query) {
    Objects.requireNonNull(query, "Query vector must not be null");
    return distance(buffer, offset, query, 0, query.length);
  }

  @Override
  public float distance(float[] bufferA, int offsetA, float[] bufferB, int offsetB, int length) {
    Objects.requireNonNull(bufferA, "Buffer A must not be null");
    Objects.requireNonNull(bufferB, "Buffer B must not be null");
    if (length < 0) {
      throw new IllegalArgumentException("Length must be non-negative: " + length);
    }
    if (offsetA < 0 || offsetA + length > bufferA.length) {
      throw new IndexOutOfBoundsException("BufferA out of bounds");
    }
    if (offsetB < 0 || offsetB + length > bufferB.length) {
      throw new IndexOutOfBoundsException("BufferB out of bounds");
    }

    int upperBound = SPECIES.loopBound(length);
    FloatVector sumVec = FloatVector.zero(SPECIES);

    int i = 0;
    for (; i < upperBound; i += SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(SPECIES, bufferA, offsetA + i);
      FloatVector vb = FloatVector.fromArray(SPECIES, bufferB, offsetB + i);
      sumVec = va.fma(vb, sumVec);
    }

    float dot = sumVec.reduceLanes(VectorOperators.ADD);

    // Scalar tail loop
    for (; i < length; i++) {
      dot += bufferA[offsetA + i] * bufferB[offsetB + i];
    }

    return Math.max(0.0f, 1.0f - dot);
  }

  @Override
  public DistanceMetric metric() {
    return DistanceMetric.COSINE;
  }
}
```

### `VectorDotProductDistance.java` (SIMD)
```java
package com.nanovector.core.distance;

import java.util.Objects;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class VectorDotProductDistance implements DistanceCalculator {

  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  @Override
  public float distance(float[] a, float[] b) {
    Objects.requireNonNull(a, "Vector 'a' must not be null");
    Objects.requireNonNull(b, "Vector 'b' must not be null");
    if (a.length != b.length) {
      throw new IllegalArgumentException("Dimension mismatch");
    }
    return distance(a, 0, b, 0, a.length);
  }

  @Override
  public float distance(float[] buffer, int offset, float[] query) {
    Objects.requireNonNull(query, "Query vector must not be null");
    return distance(buffer, offset, query, 0, query.length);
  }

  @Override
  public float distance(float[] bufferA, int offsetA, float[] bufferB, int offsetB, int length) {
    Objects.requireNonNull(bufferA, "Buffer A must not be null");
    Objects.requireNonNull(bufferB, "Buffer B must not be null");
    if (length < 0) {
      throw new IllegalArgumentException("Length must be non-negative: " + length);
    }
    if (offsetA < 0 || offsetA + length > bufferA.length) {
      throw new IndexOutOfBoundsException("BufferA out of bounds");
    }
    if (offsetB < 0 || offsetB + length > bufferB.length) {
      throw new IndexOutOfBoundsException("BufferB out of bounds");
    }

    int upperBound = SPECIES.loopBound(length);
    FloatVector sumVec = FloatVector.zero(SPECIES);

    int i = 0;
    for (; i < upperBound; i += SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(SPECIES, bufferA, offsetA + i);
      FloatVector vb = FloatVector.fromArray(SPECIES, bufferB, offsetB + i);
      sumVec = va.fma(vb, sumVec);
    }

    float dot = sumVec.reduceLanes(VectorOperators.ADD);

    // Scalar tail loop
    for (; i < length; i++) {
      dot += bufferA[offsetA + i] * bufferB[offsetB + i];
    }

    return -dot;
  }

  @Override
  public DistanceMetric metric() {
    return DistanceMetric.DOT_PRODUCT;
  }
}
```

---

## 3.4. Chỉ Mục Tham Chiếu (Ground Truth Oracle)

### `FlatIndex.java`
```java
package com.nanovector.core.index;

import com.nanovector.core.distance.DistanceCalculator;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.heap.BoundedMaxHeap;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.storage.VectorStorage;
import com.nanovector.core.util.VectorUtils;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Exact brute-force vector index ($O(N)$ sequential scan).
 *
 * <p>Serves as the <b>Ground Truth Oracle</b> for measuring recall and verifying approximate
 * nearest neighbor (ANN) indexes.
 */
public final class FlatIndex implements VectorIndex {

  private final int dimension;
  private final DistanceMetric metric;
  private final DistanceCalculator calculator;
  private final VectorStorage storage;

  public FlatIndex(int dimension, DistanceMetric metric) {
    this(dimension, metric, 1024, false);
  }

  public FlatIndex(int dimension, DistanceMetric metric, boolean useSimd) {
    this(dimension, metric, 1024, useSimd);
  }

  public FlatIndex(int dimension, DistanceMetric metric, int initialCapacity) {
    this(dimension, metric, initialCapacity, false);
  }

  public FlatIndex(int dimension, DistanceMetric metric, int initialCapacity, boolean useSimd) {
    this(dimension, metric, initialCapacity, DistanceCalculator.create(metric, useSimd));
  }

  public FlatIndex(
      int dimension, DistanceMetric metric, int initialCapacity, DistanceCalculator calculator) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    this.dimension = dimension;
    this.metric = Objects.requireNonNull(metric, "Metric must not be null");
    this.calculator = Objects.requireNonNull(calculator, "Calculator must not be null");
    this.storage = new VectorStorage(dimension, initialCapacity);
  }

  public DistanceCalculator calculator() {
    return calculator;
  }

  @Override
  public synchronized void insert(long id, float[] vector) {
    VectorUtils.checkDimension(vector, dimension);
    VectorUtils.checkFinite(vector);

    float[] effectiveVector = (metric == DistanceMetric.COSINE)
        ? VectorUtils.normalize(vector)
        : vector;

    storage.insert(id, effectiveVector);
  }

  @Override
  public List<SearchResult> searchKnn(float[] query, int k) {
    VectorUtils.checkDimension(query, dimension);
    VectorUtils.checkFinite(query);
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }

    int currentSize = storage.size();
    if (currentSize == 0) {
      return Collections.emptyList();
    }

    float[] effectiveQuery = (metric == DistanceMetric.COSINE)
        ? VectorUtils.normalize(query)
        : query;

    int heapCapacity = Math.min(k, currentSize);
    BoundedMaxHeap heap = new BoundedMaxHeap(heapCapacity);

    float[] buffer = storage.getVectorBuffer();
    for (int internalId = 0; internalId < currentSize; internalId++) {
      int offset = internalId * dimension;
      float dist = calculator.distance(buffer, offset, effectiveQuery);
      long externalId = storage.getExternalId(internalId);
      heap.offer(externalId, dist);
    }

    return heap.toSortedList();
  }

  @Override
  public int size() {
    return storage.size();
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public DistanceMetric metric() {
    return metric;
  }
}
```

---

## 3.5. Cấu Trúc Dữ Liệu & Giải Thuật HNSW

### `BoundedMaxHeap.java`
```java
package com.nanovector.core.heap;

import com.nanovector.core.model.SearchResult;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class BoundedMaxHeap {

  private final int capacity;
  private final long[] ids;
  private final float[] distances;
  private int size;

  public BoundedMaxHeap(int capacity) {
    if (capacity <= 0) {
      throw new IllegalArgumentException("Capacity must be positive: " + capacity);
    }
    this.capacity = capacity;
    this.ids = new long[capacity];
    this.distances = new float[capacity];
    this.size = 0;
  }

  public boolean offer(long id, float distance) {
    if (size < capacity) {
      ids[size] = id;
      distances[size] = distance;
      siftUp(size);
      size++;
      return true;
    }

    if (isWorse(distances[0], ids[0], distance, id)) {
      ids[0] = id;
      distances[0] = distance;
      siftDown(0);
      return true;
    }

    return false;
  }

  private boolean isWorse(float d1, long id1, float d2, long id2) {
    int cmp = Float.compare(d1, d2);
    if (cmp != 0) {
      return cmp > 0;
    }
    return id1 > id2;
  }

  private void siftUp(int index) {
    int current = index;
    long targetId = ids[current];
    float targetDistance = distances[current];

    while (current > 0) {
      int parent = (current - 1) >>> 1;
      if (isWorse(targetDistance, targetId, distances[parent], ids[parent])) {
        ids[current] = ids[parent];
        distances[current] = distances[parent];
        current = parent;
      } else {
        break;
      }
    }
    ids[current] = targetId;
    distances[current] = targetDistance;
  }

  private void siftDown(int index) {
    int current = index;
    long targetId = ids[current];
    float targetDistance = distances[current];
    int half = size >>> 1;

    while (current < half) {
      int child = (current << 1) + 1;
      int right = child + 1;

      if (right < size && isWorse(distances[right], ids[right], distances[child], ids[child])) {
        child = right;
      }

      if (isWorse(distances[child], ids[child], targetDistance, targetId)) {
        ids[current] = ids[child];
        distances[current] = distances[child];
        current = child;
      } else {
        break;
      }
    }
    ids[current] = targetId;
    distances[current] = targetDistance;
  }

  public List<SearchResult> toSortedList() {
    if (size == 0) {
      return Collections.emptyList();
    }

    SearchResult[] temp = new SearchResult[size];
    int count = size;

    for (int i = count - 1; i >= 0; i--) {
      temp[i] = new SearchResult(ids[0], distances[0]);
      size--;
      if (size > 0) {
        ids[0] = ids[size];
        distances[0] = distances[size];
        siftDown(0);
      }
    }

    List<SearchResult> result = new ArrayList<>(count);
    Collections.addAll(result, temp);
    return result;
  }

  public int size() {
    return size;
  }

  public int capacity() {
    return capacity;
  }

  public float peekMaxDistance() {
    if (size == 0) {
      throw new IllegalStateException("Heap is empty");
    }
    return distances[0];
  }
}
```

### `EpochVisitedSet.java`
```java
package com.nanovector.core.hnsw;

import java.util.Arrays;

public final class EpochVisitedSet {

  private static final int DEFAULT_INITIAL_CAPACITY = 1024;

  private int[] visitedEpoch;
  private int currentEpoch;

  public EpochVisitedSet() {
    this(DEFAULT_INITIAL_CAPACITY);
  }

  public EpochVisitedSet(int initialCapacity) {
    if (initialCapacity <= 0) {
      throw new IllegalArgumentException("Initial capacity must be positive: " + initialCapacity);
    }
    this.visitedEpoch = new int[initialCapacity];
    this.currentEpoch = 1;
  }

  public boolean isVisited(int internalId) {
    if (internalId < 0) {
      throw new IndexOutOfBoundsException("Internal ID cannot be negative: " + internalId);
    }
    if (internalId >= visitedEpoch.length) {
      return false;
    }
    return visitedEpoch[internalId] == currentEpoch;
  }

  public void markVisited(int internalId) {
    if (internalId < 0) {
      throw new IndexOutOfBoundsException("Internal ID cannot be negative: " + internalId);
    }
    ensureCapacity(internalId + 1);
    visitedEpoch[internalId] = currentEpoch;
  }

  public void nextEpoch() {
    if (currentEpoch == Integer.MAX_VALUE) {
      Arrays.fill(visitedEpoch, 0);
      currentEpoch = 1;
    } else {
      currentEpoch++;
    }
  }

  public void ensureCapacity(int minCapacity) {
    if (minCapacity > visitedEpoch.length) {
      int newCapacity = Math.max(minCapacity, visitedEpoch.length * 2);
      visitedEpoch = Arrays.copyOf(visitedEpoch, newCapacity);
    }
  }

  public int capacity() {
    return visitedEpoch.length;
  }

  public int currentEpoch() {
    return currentEpoch;
  }
}
```

### `NeighborSelector.java`
```java
package com.nanovector.core.hnsw;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public final class NeighborSelector {

  @FunctionalInterface
  public interface NodeDistanceEvaluator {
    float distance(int nodeA, int nodeB);
  }

  public record Candidate(int id, float distance) implements Comparable<Candidate> {
    @Override
    public int compareTo(Candidate other) {
      Objects.requireNonNull(other, "Cannot compare with null candidate");
      int cmp = Float.compare(this.distance, other.distance);
      if (cmp != 0) {
        return cmp;
      }
      return Integer.compare(this.id, other.id);
    }
  }

  private NeighborSelector() {}

  public static int[] selectNeighbors(
      List<Candidate> candidates, int maxDegree, NodeDistanceEvaluator distanceEvaluator) {
    Objects.requireNonNull(candidates, "Candidates list must not be null");
    Objects.requireNonNull(distanceEvaluator, "DistanceEvaluator must not be null");
    if (maxDegree <= 0) {
      throw new IllegalArgumentException("maxDegree must be positive: " + maxDegree);
    }
    if (candidates.isEmpty()) {
      return new int[0];
    }

    Collections.sort(candidates);

    int[] selected = new int[maxDegree];
    int selectedCount = 0;
    List<Candidate> discarded = new ArrayList<>();

    // Phase 1: Algorithm 4 Heuristic
    for (Candidate candidate : candidates) {
      if (selectedCount >= maxDegree) {
        break;
      }

      int e = candidate.id();
      float distToTarget = candidate.distance();
      boolean isCloserToTargetThanAnyNeighbor = true;

      for (int i = 0; i < selectedCount; i++) {
        int r = selected[i];
        if (e == r) {
          isCloserToTargetThanAnyNeighbor = false;
          break;
        }
        float distToSelected = distanceEvaluator.distance(e, r);
        if (distToSelected <= distToTarget) {
          isCloserToTargetThanAnyNeighbor = false;
          break;
        }
      }

      if (isCloserToTargetThanAnyNeighbor) {
        selected[selectedCount++] = e;
      } else {
        discarded.add(candidate);
      }
    }

    // Phase 2: NanoVector Fallback
    if (selectedCount < maxDegree && !discarded.isEmpty()) {
      for (Candidate fallback : discarded) {
        if (selectedCount >= maxDegree) {
          break;
        }
        boolean alreadySelected = false;
        for (int i = 0; i < selectedCount; i++) {
          if (selected[i] == fallback.id()) {
            alreadySelected = true;
            break;
          }
        }
        if (!alreadySelected) {
          selected[selectedCount++] = fallback.id();
        }
      }
    }

    return selectedCount == maxDegree ? selected : Arrays.copyOf(selected, selectedCount);
  }
}
```

### `HnswGraph.java`
```java
package com.nanovector.core.hnsw;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;

public final class HnswGraph {

  @FunctionalInterface
  public interface DistanceToQuery {
    float distance(int internalId);
  }

  private final HnswConfig config;
  private final List<HnswNode> nodes;
  private int entryPointId;
  private int maxLevel;

  public HnswGraph(HnswConfig config) {
    this.config = Objects.requireNonNull(config, "HnswConfig must not be null");
    this.nodes = new ArrayList<>();
    this.entryPointId = -1;
    this.maxLevel = -1;
  }

  public void addNode(HnswNode node) {
    Objects.requireNonNull(node, "Node must not be null");
    if (node.internalId() != nodes.size()) {
      throw new IllegalArgumentException(
          "Node internalId must be " + nodes.size() + ", but got: " + node.internalId());
    }
    nodes.add(node);

    if (entryPointId == -1) {
      entryPointId = node.internalId();
      maxLevel = node.maxLevel();
    }
  }

  public void setEntryPoint(int entryPointId, int maxLevel) {
    if (entryPointId < 0 || entryPointId >= nodes.size()) {
      throw new IndexOutOfBoundsException("Entry point ID out of bounds");
    }
    this.entryPointId = entryPointId;
    this.maxLevel = maxLevel;
  }

  public HnswNode getNode(int internalId) {
    if (internalId < 0 || internalId >= nodes.size()) {
      throw new IndexOutOfBoundsException("Internal ID out of bounds");
    }
    return nodes.get(internalId);
  }

  public List<NeighborSelector.Candidate> searchLayer(
      DistanceToQuery distanceToQuery,
      List<Integer> entryPoints,
      int ef,
      int layer,
      EpochVisitedSet visitedSet) {
    Objects.requireNonNull(distanceToQuery, "DistanceToQuery must not be null");
    Objects.requireNonNull(entryPoints, "Entry points list must not be null");
    Objects.requireNonNull(visitedSet, "EpochVisitedSet must not be null");
    if (ef <= 0) {
      throw new IllegalArgumentException("ef must be positive: " + ef);
    }
    if (entryPoints.isEmpty()) {
      return List.of();
    }

    visitedSet.nextEpoch();

    PriorityQueue<NeighborSelector.Candidate> candidates =
        new PriorityQueue<>(Comparator.naturalOrder()); // Min-Heap
    PriorityQueue<NeighborSelector.Candidate> results =
        new PriorityQueue<>(Comparator.reverseOrder()); // Max-Heap

    for (int ep : entryPoints) {
      visitedSet.markVisited(ep);
      float dist = distanceToQuery.distance(ep);
      NeighborSelector.Candidate cand = new NeighborSelector.Candidate(ep, dist);
      candidates.offer(cand);
      results.offer(cand);
    }

    while (!candidates.isEmpty()) {
      NeighborSelector.Candidate current = candidates.poll();
      NeighborSelector.Candidate furthestResult = results.peek();

      if (current.distance() > furthestResult.distance()) {
        break;
      }

      HnswNode currNode = nodes.get(current.id());
      int[] neighbors = currNode.getNeighbors(layer);

      for (int neighborId : neighbors) {
        if (!visitedSet.isVisited(neighborId)) {
          visitedSet.markVisited(neighborId);

          furthestResult = results.peek();
          float dist = distanceToQuery.distance(neighborId);

          if (dist < furthestResult.distance() || results.size() < ef) {
            NeighborSelector.Candidate nextCand =
                new NeighborSelector.Candidate(neighborId, dist);
            candidates.offer(nextCand);
            results.offer(nextCand);

            if (results.size() > ef) {
              results.poll();
            }
          }
        }
      }
    }

    List<NeighborSelector.Candidate> sorted = new ArrayList<>(results);
    sorted.sort(Comparator.naturalOrder());
    return sorted;
  }

  public int greedyClosest(DistanceToQuery distanceToQuery, int currentEp, int layer) {
    Objects.requireNonNull(distanceToQuery, "DistanceToQuery must not be null");
    int curr = currentEp;
    float currDist = distanceToQuery.distance(curr);

    boolean changed = true;
    while (changed) {
      changed = false;
      int[] neighbors = nodes.get(curr).getNeighbors(layer);
      for (int neighborId : neighbors) {
        float neighborDist = distanceToQuery.distance(neighborId);
        if (neighborDist < currDist) {
          currDist = neighborDist;
          curr = neighborId;
          changed = true;
        }
      }
    }
    return curr;
  }

  public void connect(
      int newNodeId,
      int layer,
      List<NeighborSelector.Candidate> candidateList,
      NeighborSelector.NodeDistanceEvaluator distanceEvaluator) {
    HnswNode newNode = nodes.get(newNodeId);
    int maxDegree = (layer == 0) ? config.m0() : config.m();

    int[] selectedNeighbors =
        NeighborSelector.selectNeighbors(
            new ArrayList<>(candidateList), maxDegree, distanceEvaluator);

    for (int neighborId : selectedNeighbors) {
      newNode.addNeighbor(layer, neighborId);
      nodes.get(neighborId).addNeighbor(layer, newNodeId);
    }

    for (int neighborId : selectedNeighbors) {
      HnswNode neighborNode = nodes.get(neighborId);
      if (neighborNode.degree(layer) > maxDegree) {
        prune(neighborNode.internalId(), layer, maxDegree, distanceEvaluator);
      }
    }
  }

  public void prune(
      int targetNodeId,
      int layer,
      int maxDegree,
      NeighborSelector.NodeDistanceEvaluator distanceEvaluator) {
    HnswNode targetNode = nodes.get(targetNodeId);
    int[] currentNeighbors = targetNode.getNeighbors(layer);
    if (currentNeighbors.length <= maxDegree) {
      return;
    }

    List<NeighborSelector.Candidate> candidates = new ArrayList<>(currentNeighbors.length);
    for (int n : currentNeighbors) {
      float dist = distanceEvaluator.distance(targetNodeId, n);
      candidates.add(new NeighborSelector.Candidate(n, dist));
    }

    int[] selected =
        NeighborSelector.selectNeighbors(candidates, maxDegree, distanceEvaluator);

    int[] removedNeighbors = new int[currentNeighbors.length - selected.length];
    int remIdx = 0;
    for (int oldN : currentNeighbors) {
      boolean kept = false;
      for (int s : selected) {
        if (s == oldN) {
          kept = true;
          break;
        }
      }
      if (!kept) {
        removedNeighbors[remIdx++] = oldN;
      }
    }

    targetNode.setNeighbors(layer, selected);

    for (int removed : removedNeighbors) {
      nodes.get(removed).removeNeighbor(layer, targetNodeId);
    }
  }

  public int entryPointId() {
    return entryPointId;
  }

  public int maxLevel() {
    return maxLevel;
  }

  public int size() {
    return nodes.size();
  }

  public HnswConfig config() {
    return config;
  }
}
```

---

## 3.6. Bộ Điều Phối Chỉ Mục HNSW

### `HnswIndex.java`
```java
package com.nanovector.core.index;

import com.nanovector.core.distance.DistanceCalculator;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.EpochVisitedSet;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.hnsw.LevelGenerator;
import com.nanovector.core.hnsw.NeighborSelector;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.storage.VectorStorage;
import com.nanovector.core.util.VectorUtils;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public final class HnswIndex implements VectorIndex {

  private final int dimension;
  private final DistanceMetric metric;
  private final DistanceCalculator calculator;
  private final VectorStorage storage;
  private final HnswGraph graph;
  private final HnswConfig config;
  private final LevelGenerator levelGenerator;
  private final EpochVisitedSet visitedSet;
  private final NeighborSelector.NodeDistanceEvaluator nodeEvaluator;

  public HnswIndex(int dimension, DistanceMetric metric, HnswConfig config) {
    this(dimension, metric, config, 1024, true);
  }

  public HnswIndex(int dimension, DistanceMetric metric, HnswConfig config, boolean useSimd) {
    this(dimension, metric, config, 1024, useSimd);
  }

  public HnswIndex(int dimension, DistanceMetric metric, HnswConfig config, int initialCapacity) {
    this(dimension, metric, config, initialCapacity, true);
  }

  public HnswIndex(
      int dimension,
      DistanceMetric metric,
      HnswConfig config,
      int initialCapacity,
      boolean useSimd) {
    this(dimension, metric, config, initialCapacity, DistanceCalculator.create(metric, useSimd));
  }

  public HnswIndex(
      int dimension,
      DistanceMetric metric,
      HnswConfig config,
      int initialCapacity,
      DistanceCalculator calculator) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    this.dimension = dimension;
    this.metric = Objects.requireNonNull(metric, "Metric must not be null");
    this.config = Objects.requireNonNull(config, "HnswConfig must not be null");
    this.calculator = Objects.requireNonNull(calculator, "Calculator must not be null");
    this.storage = new VectorStorage(dimension, initialCapacity);
    this.graph = new HnswGraph(config);
    this.levelGenerator = new LevelGenerator(config);
    this.visitedSet = new EpochVisitedSet(initialCapacity);
    this.nodeEvaluator =
        (a, b) ->
            this.calculator.distance(
                this.storage.getVectorBuffer(),
                a * dimension,
                this.storage.getVectorBuffer(),
                b * dimension,
                dimension);
  }

  @Override
  public synchronized void insert(long id, float[] vector) {
    VectorUtils.checkDimension(vector, dimension);
    VectorUtils.checkFinite(vector);

    float[] effectiveVector = (metric == DistanceMetric.COSINE)
        ? VectorUtils.normalize(vector)
        : vector;

    int newNodeId = storage.insert(id, effectiveVector);
    int nodeLevel = levelGenerator.assignLevel();
    HnswNode newNode = new HnswNode(newNodeId, nodeLevel);
    visitedSet.ensureCapacity(newNodeId + 1);

    if (graph.size() == 0) {
      graph.addNode(newNode);
      return;
    }

    HnswGraph.DistanceToQuery distToNewNode = buildDistanceToNode(newNodeId);
    int currObj = graph.entryPointId();
    int graphMaxLevel = graph.maxLevel();

    // 1. Greedy routing top-down to nodeLevel + 1
    for (int l = graphMaxLevel; l > nodeLevel; l--) {
      currObj = graph.greedyClosest(distToNewNode, currObj, l);
    }

    // 2. Connect from min(nodeLevel, graphMaxLevel) down to 0
    List<Integer> entryPoints = new ArrayList<>();
    entryPoints.add(currObj);

    int topConnectLevel = Math.min(nodeLevel, graphMaxLevel);
    graph.addNode(newNode);

    for (int l = topConnectLevel; l >= 0; l--) {
      List<NeighborSelector.Candidate> candidates =
          graph.searchLayer(distToNewNode, entryPoints, config.efConstruction(), l, visitedSet);

      graph.connect(newNodeId, l, candidates, nodeEvaluator);

      entryPoints.clear();
      for (NeighborSelector.Candidate c : candidates) {
        entryPoints.add(c.id());
      }
    }

    if (nodeLevel > graphMaxLevel) {
      graph.setEntryPoint(newNodeId, nodeLevel);
    }
  }

  @Override
  public List<SearchResult> searchKnn(float[] query, int k) {
    return searchKnn(query, k, config.efSearch());
  }

  public List<SearchResult> searchKnn(float[] query, int k, int efSearch) {
    VectorUtils.checkDimension(query, dimension);
    VectorUtils.checkFinite(query);
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    if (efSearch <= 0) {
      throw new IllegalArgumentException("efSearch must be positive: " + efSearch);
    }

    if (graph.size() == 0) {
      return Collections.emptyList();
    }

    float[] effectiveQuery = (metric == DistanceMetric.COSINE)
        ? VectorUtils.normalize(query)
        : query;

    HnswGraph.DistanceToQuery distToQuery = buildDistanceToQuery(effectiveQuery);
    int currObj = graph.entryPointId();
    int graphMaxLevel = graph.maxLevel();

    // Top-down greedy routing
    for (int l = graphMaxLevel; l > 0; l--) {
      currObj = graph.greedyClosest(distToQuery, currObj, l);
    }

    // Search at layer 0
    List<Integer> entryPoints = List.of(currObj);
    int ef = Math.max(k, efSearch);
    List<NeighborSelector.Candidate> candidates =
        graph.searchLayer(distToQuery, entryPoints, ef, 0, visitedSet);

    int resultCount = Math.min(k, candidates.size());
    List<SearchResult> results = new ArrayList<>(resultCount);
    for (int i = 0; i < resultCount; i++) {
      NeighborSelector.Candidate cand = candidates.get(i);
      long extId = storage.getExternalId(cand.id());
      results.add(new SearchResult(extId, cand.distance()));
    }
    return results;
  }

  @Override
  public int size() {
    return storage.size();
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public DistanceMetric metric() {
    return metric;
  }

  public HnswGraph graph() {
    return graph;
  }

  public DistanceCalculator calculator() {
    return calculator;
  }

  private HnswGraph.DistanceToQuery buildDistanceToQuery(float[] queryVector) {
    float[] buffer = storage.getVectorBuffer();
    return internalId -> calculator.distance(buffer, internalId * dimension, queryVector);
  }

  private HnswGraph.DistanceToQuery buildDistanceToNode(int targetInternalId) {
    int targetOffset = targetInternalId * dimension;
    float[] buffer = storage.getVectorBuffer();
    return internalId ->
        calculator.distance(buffer, internalId * dimension, buffer, targetOffset, dimension);
  }
}
```

---

# 4. TÀI LIỆU HƯỚNG DẪN REVIEW CHUYÊN SÂU

Tài liệu này được sắp xếp sẵn để bạn có thể review chuyên sâu về 3 trụ cột kỹ thuật:

### 1. Kiến trúc & Data Structures
- **Tổ chức bộ nhớ (`VectorStorage`)**: Mảng primitive phẳng 1D (`float[]`), chỉ mục $offset = internalId \times dimension$. Loại bỏ hoàn toàn mảng lồng mảng (`float[][]`) và con trỏ rời rạc.
- **Ánh xạ ID 2 chiều**: Ánh xạ `externalId` (long) $\leftrightarrow$ `internalId` (int) thông qua `HashMap` và mảng `long[] externalIds`.
- **Đồ thị đa tầng (`HnswNode`, `HnswGraph`)**: `int[][] neighbors` tách biệt hoàn toàn khỏi `VectorStorage`. Graph chỉ quản lý topology thông qua `internalId`.

### 2. Thuật toán & Hiệu năng
- **Tối ưu SIMD qua Java Vector API**: Tự động nhận diện độ rộng lane qua `FloatVector.SPECIES_PREFERRED`, sử dụng `FloatVector.fma()` cho tích lũy FMA và xử lý chiều lệch bằng scalar tail loop.
- **Zero-Allocation Hot Path**: Tính toán khoảng cách buffer-to-query và buffer-to-buffer thông qua offset nguyên thủy, loại bỏ `Arrays.copyOfRange`.
- **Epoch-Based Visited Set**: `EpochVisitedSet` quản lý mảng `int[] visitedEpoch` tăng dần theo từng query, đạt $O(1)$ reset không cấp phát và không gọi `Arrays.fill`.
- **Heuristic Chọn Cạnh (Algorithm 4)**: Đảm bảo tính đa dạng góc và kết hợp fallback chống thưa đồ thị cục bộ. 100% đối xứng hai chiều và liên thông layer 0.

### 3. API & Khả năng mở rộng
- **Giao diện thống nhất**: Cả `FlatIndex` và `HnswIndex` cùng hiện thực `VectorIndex`.
- **Hợp đồng Distance Minimization**: Giá trị trả về càng nhỏ độ tương đồng càng cao, đảm bảo tính nhất quán trên cả Euclidean, Cosine và Dot Product.
- **Sẵn sàng cho Binary Persistence (Phase 4)**: Cấu trúc bộ nhớ tuyến tính của `VectorStorage` và adjacency list nguyên thủy của `HnswGraph` được thiết kế để tuần tự hóa thẳng xuống file nhị phân `.nvec` với streaming I/O và checksum CRC32C.
