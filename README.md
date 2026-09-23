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
   - Zero heap allocations during the $N$-vector distance scan.
4. **Deterministic Tie-Breaking**:
   - When distances are identical, ranking defaults to `externalId` ascending, ensuring 100% reproducible search results.
5. **Exact Ground Truth Baseline**:
   - `FlatIndex` ($O(N)$ sequential scan) acts as the exact reference oracle for measuring recall in approximate nearest neighbor (ANN) graphs.

---

## 🚀 Getting Started

### Prerequisites
- JDK 21 or higher (compiled with `--release 21`).
- Maven 3.9+ (or use the provided `.\mvn.cmd`).

### Build & Run Tests
```powershell
.\mvn.cmd clean test
```

---

## 🗺️ Roadmap & Evolutionary Milestones

- [x] **v0.1**: Multi-module setup, contiguous `VectorStorage`, distance metrics, primitive `BoundedMaxHeap`, `FlatIndex` Ground Truth Oracle (26/26 unit tests).
- [ ] **v0.2**: HNSW Core Graph (multi-layer routing, neighbor selection heuristic Algorithm 4).
- [ ] **v0.3**: Epoch-based visited set optimization.
- [ ] **v0.4**: Binary persistence (`.nvec` file format).
- [ ] **v0.5**: Rigorous benchmark suite (Recall@K vs Latency, Memory footprint per vector).
- [ ] **v0.6**: Standalone CLI & Spring Boot REST API.
- [ ] **v0.7**: Experimental SIMD acceleration via Java Vector API (`jdk.incubator.vector`).
