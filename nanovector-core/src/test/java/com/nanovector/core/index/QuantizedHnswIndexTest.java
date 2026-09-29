package com.nanovector.core.index;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.model.SearchResult;
import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class QuantizedHnswIndexTest {

  private static final HnswConfig CONFIG = HnswConfig.defaultConfig().withSeed(42L);

  private static float[] randomVector(Random rng, int dim) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = rng.nextFloat() * 2f - 1f;
    }
    return v;
  }

  @Test
  @DisplayName("searchKnn on empty index returns empty list")
  void testSearchEmptyIndex() {
    QuantizedHnswIndex index = new QuantizedHnswIndex(4, DistanceMetric.EUCLIDEAN, CONFIG);

    List<SearchResult> results = index.searchKnn(new float[] {1f, 2f, 3f, 4f}, 5);

    assertThat(results).isEmpty();
    assertThat(index.size()).isZero();
  }

  @Test
  @DisplayName("Single vector insertion and retrieval returns exact ID with near-zero distance")
  void testSingleVector() {
    QuantizedHnswIndex index = new QuantizedHnswIndex(3, DistanceMetric.EUCLIDEAN, CONFIG);
    float[] vec = {1.0f, 2.0f, 3.0f};

    index.insert(100L, vec);

    assertThat(index.size()).isEqualTo(1);

    List<SearchResult> results = index.searchKnn(vec, 1);
    assertThat(results).hasSize(1);
    assertThat(results.get(0).id()).isEqualTo(100L);
    assertThat(results.get(0).distance()).isLessThan(1e-4f);
  }

  @Test
  @DisplayName("Dimension mismatch throws IllegalArgumentException")
  void testDimensionMismatch() {
    QuantizedHnswIndex index = new QuantizedHnswIndex(3, DistanceMetric.EUCLIDEAN, CONFIG);

    assertThatThrownBy(() -> index.insert(1L, new float[] {1f, 2f}))
        .isInstanceOf(IllegalArgumentException.class);

    assertThatThrownBy(() -> index.searchKnn(new float[] {1f, 2f, 3f, 4f}, 1))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Invalid k and efSearch throw IllegalArgumentException")
  void testInvalidParameters() {
    QuantizedHnswIndex index = new QuantizedHnswIndex(2, DistanceMetric.EUCLIDEAN, CONFIG);
    index.insert(1L, new float[] {1f, 2f});

    assertThatThrownBy(() -> index.searchKnn(new float[] {1f, 2f}, 0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> index.searchKnn(new float[] {1f, 2f}, -1))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> index.searchKnn(new float[] {1f, 2f}, 1, 0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> index.searchKnn(new float[] {1f, 2f}, 1, -5))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Duplicate ID throws IllegalArgumentException")
  void testDuplicateId() {
    QuantizedHnswIndex index = new QuantizedHnswIndex(2, DistanceMetric.EUCLIDEAN, CONFIG);
    index.insert(1L, new float[] {1f, 2f});

    assertThatThrownBy(() -> index.insert(1L, new float[] {3f, 4f}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Non-EUCLIDEAN metric throws UnsupportedOperationException")
  void testUnsupportedMetric() {
    assertThatThrownBy(() -> new QuantizedHnswIndex(2, DistanceMetric.COSINE, CONFIG))
        .isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> new QuantizedHnswIndex(2, DistanceMetric.DOT_PRODUCT, CONFIG))
        .isInstanceOf(UnsupportedOperationException.class);
  }

  @Test
  @DisplayName("k larger than index size returns all inserted vectors sorted by distance")
  void testKLargerThanSize() {
    QuantizedHnswIndex index = new QuantizedHnswIndex(2, DistanceMetric.EUCLIDEAN, CONFIG);
    index.insert(1L, new float[] {0f, 0f});
    index.insert(2L, new float[] {10f, 10f});

    List<SearchResult> results = index.searchKnn(new float[] {0f, 0f}, 10);

    assertThat(results).hasSize(2);
    assertThat(results.get(0).id()).isEqualTo(1L);
    assertThat(results.get(1).id()).isEqualTo(2L);
    assertThat(results.get(0).distance()).isLessThan(results.get(1).distance());
  }

  @Test
  @DisplayName("vectorData throws UnsupportedOperationException directing to storage()")
  void testVectorDataUnsupported() {
    QuantizedHnswIndex index = new QuantizedHnswIndex(2, DistanceMetric.EUCLIDEAN, CONFIG);
    assertThatThrownBy(index::vectorData).isInstanceOf(UnsupportedOperationException.class);
    assertThat(index.storage()).isNotNull();
    assertThat(index.graph()).isNotNull();
  }

  @Test
  @DisplayName("Deterministic seed produces identical graph topology and search results")
  void testDeterministicSeed() {
    int dim = 16;
    int n = 50;
    long seed = 12345L;

    QuantizedHnswIndex index1 =
        new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, CONFIG.withSeed(seed));
    QuantizedHnswIndex index2 =
        new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, CONFIG.withSeed(seed));

    Random rng1 = new Random(seed);
    Random rng2 = new Random(seed);

    for (int i = 0; i < n; i++) {
      float[] v1 = randomVector(rng1, dim);
      float[] v2 = randomVector(rng2, dim);
      index1.insert(i, v1);
      index2.insert(i, v2);
    }

    assertThat(index1.graph().maxLevel()).isEqualTo(index2.graph().maxLevel());
    assertThat(index1.graph().entryPointId()).isEqualTo(index2.graph().entryPointId());

    for (int i = 0; i < n; i++) {
      HnswNode node1 = index1.graph().getNode(i);
      HnswNode node2 = index2.graph().getNode(i);
      assertThat(node1.maxLevel()).isEqualTo(node2.maxLevel());
      for (int l = 0; l <= node1.maxLevel(); l++) {
        assertThat(node1.getNeighbors(l)).containsExactly(node2.getNeighbors(l));
      }
    }

    float[] query = randomVector(new Random(999L), dim);
    List<SearchResult> res1 = index1.searchKnn(query, 5);
    List<SearchResult> res2 = index2.searchKnn(query, 5);

    assertThat(res1).hasSize(5);
    assertThat(res2).hasSize(5);
    for (int i = 0; i < 5; i++) {
      assertThat(res1.get(i).id()).isEqualTo(res2.get(i).id());
      assertThat(res1.get(i).distance()).isEqualTo(res2.get(i).distance());
    }
  }

  @Test
  @DisplayName("Degree constraints: Layer 0 <= M0, Layer > 0 <= M")
  void testDegreeConstraints() {
    int dim = 16;
    int n = 200;
    QuantizedHnswIndex index = new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, CONFIG);
    Random rng = new Random(42L);

    for (int i = 0; i < n; i++) {
      index.insert(i, randomVector(rng, dim));
    }

    HnswGraph graph = index.graph();
    HnswConfig cfg = index.config();

    for (int i = 0; i < graph.size(); i++) {
      HnswNode node = graph.getNode(i);
      assertThat(node.degree(0))
          .as("Node %d degree at layer 0 must be <= M0 (%d)", i, cfg.m0())
          .isLessThanOrEqualTo(cfg.m0());

      for (int l = 1; l <= node.maxLevel(); l++) {
        assertThat(node.degree(l))
            .as("Node %d degree at layer %d must be <= M (%d)", i, l, cfg.m())
            .isLessThanOrEqualTo(cfg.m());
      }
    }
  }

  @Test
  @DisplayName("Layer 0 has zero isolated nodes: min(degree) >= 1")
  void testLayer0ZeroIsolatedNodes() {
    int dim = 16;
    int n = 200;
    QuantizedHnswIndex index = new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, CONFIG);
    Random rng = new Random(42L);

    for (int i = 0; i < n; i++) {
      index.insert(i, randomVector(rng, dim));
    }

    HnswGraph graph = index.graph();
    for (int i = 0; i < graph.size(); i++) {
      assertThat(graph.getNode(i).degree(0))
          .as("Node %d in Layer 0 must have at least 1 connection", i)
          .isGreaterThanOrEqualTo(1);
    }
  }

  @Test
  @DisplayName("Layer 0 connectivity via BFS: visits 100% of nodes")
  void testLayer0ConnectivityViaBFS() {
    int dim = 16;
    int n = 200;
    QuantizedHnswIndex index = new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, CONFIG);
    Random rng = new Random(42L);

    for (int i = 0; i < n; i++) {
      index.insert(i, randomVector(rng, dim));
    }

    HnswGraph graph = index.graph();
    Set<Integer> visited = new HashSet<>();
    Queue<Integer> queue = new ArrayDeque<>();

    queue.add(0);
    visited.add(0);

    while (!queue.isEmpty()) {
      int current = queue.poll();
      for (int neighbor : graph.getNode(current).getNeighbors(0)) {
        if (visited.add(neighbor)) {
          queue.add(neighbor);
        }
      }
    }

    assertThat(visited).as("BFS on Layer 0 should visit all %d nodes", n).hasSize(n);
  }

  @Test
  @DisplayName("Tail dimensions (D=65, D=33) run without errors or bounds violations")
  void testTailDimensions() {
    int[] tailDims = {33, 65, 127};
    Random rng = new Random(777L);

    for (int dim : tailDims) {
      QuantizedHnswIndex index =
          new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, CONFIG, true);
      for (int i = 0; i < 30; i++) {
        index.insert(i, randomVector(rng, dim));
      }

      float[] query = randomVector(rng, dim);
      List<SearchResult> results = index.searchKnn(query, 5);

      assertThat(results).hasSize(5);
      for (int i = 0; i < 4; i++) {
        assertThat(results.get(i).distance()).isLessThanOrEqualTo(results.get(i + 1).distance());
      }
    }
  }

  @Test
  @DisplayName("Scalar vs SIMD search returns identical top-K ranking")
  void testScalarVsSimdParity() {
    int dim = 128;
    int n = 100;
    Random rng = new Random(123L);

    QuantizedHnswIndex simded = new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, CONFIG, true);
    for (int i = 0; i < n; i++) {
      simded.insert(i, randomVector(rng, dim));
    }

    // Wrap the same storage and graph in a scalar index
    QuantizedHnswIndex scalar =
        new QuantizedHnswIndex(
            dim,
            DistanceMetric.EUCLIDEAN,
            CONFIG,
            simded.storage(),
            simded.graph(),
            com.nanovector.core.quantization.QuantizedEuclideanDistance.create(false));

    Random queryRng = new Random(456L);
    for (int q = 0; q < 10; q++) {
      float[] query = randomVector(queryRng, dim);
      List<SearchResult> simdResults = simded.searchKnn(query, 10);
      List<SearchResult> scalarResults = scalar.searchKnn(query, 10);

      assertThat(simdResults).hasSameSizeAs(scalarResults);
      for (int i = 0; i < simdResults.size(); i++) {
        assertThat(simdResults.get(i).id()).isEqualTo(scalarResults.get(i).id());
        assertThat(simdResults.get(i).distance())
            .isCloseTo(scalarResults.get(i).distance(), org.assertj.core.data.Offset.offset(1e-4f));
      }
    }
  }

  @Test
  @DisplayName("fromFp32 creates valid QuantizedHnswIndex with 100% graph structure sharing")
  void testFromFp32HybridConstruction() {
    int dim = 32;
    int n = 100;
    Random rng = new Random(42L);

    HnswIndex fp32Index = new HnswIndex(dim, DistanceMetric.EUCLIDEAN, CONFIG);
    for (int i = 0; i < n; i++) {
      fp32Index.insert(i * 10L, randomVector(rng, dim));
    }

    QuantizedHnswIndex hybridIndex = QuantizedHnswIndex.fromFp32(fp32Index);

    assertThat(hybridIndex.size()).isEqualTo(fp32Index.size());
    assertThat(hybridIndex.graph()).isSameAs(fp32Index.graph());
    assertThat(hybridIndex.dimension()).isEqualTo(dim);

    float[] query = randomVector(new Random(999L), dim);
    List<SearchResult> fp32Res = fp32Index.searchKnn(query, 10);
    List<SearchResult> hybridRes = hybridIndex.searchKnn(query, 10);

    assertThat(hybridRes).hasSize(10);
    // Overlap should be very high
    Set<Long> fp32Ids = new HashSet<>();
    for (SearchResult r : fp32Res) fp32Ids.add(r.id());

    int overlap = 0;
    for (SearchResult r : hybridRes) {
      if (fp32Ids.contains(r.id())) overlap++;
    }
    assertThat(overlap).isGreaterThanOrEqualTo(8); // at least 80% overlap at 100 vectors
  }

  @Test
  @DisplayName("Sanity Recall check against FP32 Flat oracle on 1,000 vectors")
  void testSanityRecallAgainstFlatOracle() {
    int dim = 128;
    int n = 1000;
    int k = 10;
    Random rng = new Random(42L);

    FlatIndex flatOracle = new FlatIndex(dim, DistanceMetric.EUCLIDEAN);
    QuantizedHnswIndex hnswIndex = new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, CONFIG);

    for (int i = 0; i < n; i++) {
      float[] v = randomVector(rng, dim);
      flatOracle.insert(i, v);
      hnswIndex.insert(i, v);
    }

    Random queryRng = new Random(12345L);
    int numQueries = 20;
    int totalMatches = 0;

    for (int q = 0; q < numQueries; q++) {
      float[] query = randomVector(queryRng, dim);
      List<SearchResult> truth = flatOracle.searchKnn(query, k);
      List<SearchResult> approx = hnswIndex.searchKnn(query, k, 100);

      Set<Long> truthSet = new HashSet<>();
      for (SearchResult r : truth) {
        truthSet.add(r.id());
      }

      for (SearchResult r : approx) {
        if (truthSet.contains(r.id())) {
          totalMatches++;
        }
      }
    }

    double recall = (double) totalMatches / (numQueries * k);
    // At N=1,000, D=128, efSearch=100, SQ8 HNSW recall should easily exceed 90%
    assertThat(recall).isGreaterThan(0.90);
  }

  @Test
  @DisplayName("Verify searchKnnWithRerank evaluates exact FP32 distances and re-sorts candidates")
  void testSearchKnnWithRerank() {
    int dim = 32;
    int n = 100;
    int k = 5;
    int efSearch = 20;
    Random rng = new Random(42L);

    com.nanovector.core.storage.VectorStorage rawStorage =
        new com.nanovector.core.storage.VectorStorage(dim, n);
    QuantizedHnswIndex index = new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, CONFIG);

    for (int i = 0; i < n; i++) {
      float[] v = randomVector(rng, dim);
      rawStorage.insert(i, v);
      index.insert(i, v);
    }

    float[] query = randomVector(new Random(999L), dim);

    // Standard search
    List<SearchResult> stdResults = index.searchKnn(query, k, efSearch);
    assertThat(stdResults).hasSize(k);

    // Re-ranked search
    List<SearchResult> rerankedResults = index.searchKnnWithRerank(query, k, efSearch, rawStorage);
    assertThat(rerankedResults).hasSize(k);

    // Distance must be strictly ascending
    for (int i = 0; i < k - 1; i++) {
      assertThat(rerankedResults.get(i).distance())
          .isLessThanOrEqualTo(rerankedResults.get(i + 1).distance());
    }

    // Verify exact distances
    float[] rawBuffer = rawStorage.vectorBuffer();
    for (SearchResult r : rerankedResults) {
      int id = (int) r.id();
      float expectedDist = 0.0f;
      for (int d = 0; d < dim; d++) {
        float diff = query[d] - rawBuffer[id * dim + d];
        expectedDist += diff * diff;
      }
      assertThat(r.distance()).isCloseTo(expectedDist, org.assertj.core.data.Offset.offset(1e-5f));
    }

    // Candidate search
    List<SearchResult> candidates = index.searchKnnCandidates(query, efSearch);
    assertThat(candidates).isNotEmpty();
    assertThat(candidates.size()).isLessThanOrEqualTo(efSearch);

    // Error handling
    assertThatThrownBy(() -> index.searchKnnWithRerank(query, 0, efSearch, rawStorage))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> index.searchKnnWithRerank(query, k, 0, rawStorage))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                index.searchKnnWithRerank(
                    query, k, efSearch, (com.nanovector.core.storage.VectorStorage) null))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> index.searchKnnCandidates(query, 0))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
