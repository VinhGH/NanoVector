package com.nanovector.core.index;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.model.SearchResult;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class HnswIndexTest {

  private static final HnswConfig CONFIG = HnswConfig.defaultConfig().withSeed(42L);

  @Test
  @DisplayName("searchKnn on empty index returns empty list")
  void testSearchEmptyIndex() {
    HnswIndex index = new HnswIndex(4, DistanceMetric.EUCLIDEAN, CONFIG);

    List<SearchResult> results = index.searchKnn(new float[] {1f, 2f, 3f, 4f}, 5);

    assertThat(results).isEmpty();
    assertThat(index.size()).isZero();
  }

  @Test
  @DisplayName("Single vector insertion and retrieval returns exact vector with distance 0")
  void testSingleVector() {
    HnswIndex index = new HnswIndex(3, DistanceMetric.EUCLIDEAN, CONFIG);
    float[] vec = {1.0f, 2.0f, 3.0f};

    index.insert(100L, vec);

    assertThat(index.size()).isEqualTo(1);

    List<SearchResult> results = index.searchKnn(vec, 1);
    assertThat(results).hasSize(1);
    assertThat(results.get(0).id()).isEqualTo(100L);
    assertThat(results.get(0).distance()).isEqualTo(0.0f);
  }

  @Test
  @DisplayName("Dimension mismatch throws IllegalArgumentException")
  void testDimensionMismatch() {
    HnswIndex index = new HnswIndex(3, DistanceMetric.EUCLIDEAN, CONFIG);

    assertThatThrownBy(() -> index.insert(1L, new float[] {1f, 2f}))
        .isInstanceOf(IllegalArgumentException.class);

    assertThatThrownBy(() -> index.searchKnn(new float[] {1f, 2f, 3f, 4f}, 1))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Invalid k throws IllegalArgumentException")
  void testInvalidK() {
    HnswIndex index = new HnswIndex(2, DistanceMetric.EUCLIDEAN, CONFIG);
    index.insert(1L, new float[] {1f, 2f});

    assertThatThrownBy(() -> index.searchKnn(new float[] {1f, 2f}, 0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> index.searchKnn(new float[] {1f, 2f}, -1))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Duplicate ID throws IllegalArgumentException")
  void testDuplicateId() {
    HnswIndex index = new HnswIndex(2, DistanceMetric.EUCLIDEAN, CONFIG);
    index.insert(1L, new float[] {1f, 2f});

    assertThatThrownBy(() -> index.insert(1L, new float[] {3f, 4f}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("k larger than index size returns all inserted vectors")
  void testKLargerThanSize() {
    HnswIndex index = new HnswIndex(2, DistanceMetric.EUCLIDEAN, CONFIG);
    index.insert(1L, new float[] {0f, 0f});
    index.insert(2L, new float[] {1f, 1f});

    List<SearchResult> results = index.searchKnn(new float[] {0f, 0f}, 10);

    assertThat(results).hasSize(2);
    assertThat(results.get(0).id()).isEqualTo(1L);
    assertThat(results.get(1).id()).isEqualTo(2L);
  }

  @Test
  @DisplayName("Cosine similarity auto-normalizes vectors")
  void testCosineMetric() {
    HnswIndex index = new HnswIndex(2, DistanceMetric.COSINE, CONFIG);
    index.insert(1L, new float[] {0f, 5f}); // normalized: [0, 1]
    index.insert(2L, new float[] {10f, 0f}); // normalized: [1, 0]

    // Query along y-axis: [0, 2] -> should match ID 1 with distance ~0
    List<SearchResult> results = index.searchKnn(new float[] {0f, 2f}, 2);

    assertThat(results).hasSize(2);
    assertThat(results.get(0).id()).isEqualTo(1L);
    assertThat(results.get(0).distance())
        .isCloseTo(0.0f, org.assertj.core.data.Offset.offset(1e-5f));
  }

  @Test
  @DisplayName("Dot product metric ranks higher dot product as lower distance")
  void testDotProductMetric() {
    HnswIndex index = new HnswIndex(2, DistanceMetric.DOT_PRODUCT, CONFIG);
    index.insert(1L, new float[] {1f, 0f});
    index.insert(2L, new float[] {5f, 0f});

    // Query [1, 0]: dot(2, q) = 5 -> distance -5; dot(1, q) = 1 -> distance -1
    List<SearchResult> results = index.searchKnn(new float[] {1f, 0f}, 2);

    assertThat(results).hasSize(2);
    assertThat(results.get(0).id()).isEqualTo(2L);
    assertThat(results.get(0).distance()).isEqualTo(-5.0f);
  }

  @Test
  @DisplayName("Multi-vector search finds nearest neighbors in 2D space")
  void testMultiVectorSearch() {
    HnswIndex index = new HnswIndex(2, DistanceMetric.EUCLIDEAN, CONFIG);

    index.insert(1L, new float[] {0.0f, 0.0f});
    index.insert(2L, new float[] {1.0f, 0.0f});
    index.insert(3L, new float[] {0.0f, 1.0f});
    index.insert(4L, new float[] {10.0f, 10.0f});
    index.insert(5L, new float[] {10.0f, 11.0f});

    // Query near origin: (0.1, 0.1)
    List<SearchResult> results = index.searchKnn(new float[] {0.1f, 0.1f}, 3);

    assertThat(results).hasSize(3);
    assertThat(results.get(0).id()).isEqualTo(1L);
    assertThat(results).extracting(SearchResult::id).containsExactlyInAnyOrder(1L, 2L, 3L);
  }

  @Test
  @DisplayName(
      "searchKnn with custom efSearch returns valid results and enforces positive efSearch")
  void testSearchKnnWithCustomEfSearch() {
    HnswIndex index = new HnswIndex(2, DistanceMetric.EUCLIDEAN, CONFIG);
    index.insert(1L, new float[] {0.0f, 0.0f});
    index.insert(2L, new float[] {1.0f, 0.0f});

    List<SearchResult> results = index.searchKnn(new float[] {0.0f, 0.0f}, 1, 10);
    assertThat(results).hasSize(1);
    assertThat(results.get(0).id()).isEqualTo(1L);

    assertThatThrownBy(() -> index.searchKnn(new float[] {0f, 0f}, 1, 0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> index.searchKnn(new float[] {0f, 0f}, 1, -1))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("HnswIndex defaults to SIMD and supports explicit scalar mode")
  void testHnswIndexSimdAndScalarModes() {
    HnswIndex defaultHnsw = new HnswIndex(4, DistanceMetric.EUCLIDEAN, CONFIG);
    assertThat(defaultHnsw.calculator())
        .isInstanceOf(com.nanovector.core.distance.VectorEuclideanDistance.class);

    HnswIndex scalarHnsw = new HnswIndex(4, DistanceMetric.EUCLIDEAN, CONFIG, false);
    assertThat(scalarHnsw.calculator())
        .isInstanceOf(com.nanovector.core.distance.ScalarEuclideanDistance.class);

    float[] v1 = {1.0f, 0.0f, 0.0f, 0.0f};
    float[] v2 = {0.0f, 1.0f, 0.0f, 0.0f};
    defaultHnsw.insert(1L, v1);
    defaultHnsw.insert(2L, v2);
    scalarHnsw.insert(1L, v1);
    scalarHnsw.insert(2L, v2);

    float[] query = {1.0f, 0.1f, 0.0f, 0.0f};
    List<SearchResult> defaultRes = defaultHnsw.searchKnn(query, 2);
    List<SearchResult> scalarRes = scalarHnsw.searchKnn(query, 2);

    assertThat(defaultRes).hasSize(2);
    assertThat(scalarRes).hasSize(2);
    assertThat(defaultRes.get(0).id()).isEqualTo(scalarRes.get(0).id()).isEqualTo(1L);
    assertThat(defaultRes.get(0).distance())
        .isCloseTo(scalarRes.get(0).distance(), org.assertj.core.data.Offset.offset(1e-5f));
  }
}
