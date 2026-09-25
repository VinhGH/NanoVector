package com.nanovector.core.index;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.offset;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.distance.ScalarEuclideanDistance;
import com.nanovector.core.model.SearchResult;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class FlatIndexTest {

  @Test
  @DisplayName("2D Grid: Finds closest points geometrically")
  void test2DPointGrid() {
    FlatIndex index = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    index.insert(1L, new float[] {0.0f, 0.0f});
    index.insert(2L, new float[] {1.0f, 1.0f});
    index.insert(3L, new float[] {5.0f, 5.0f});
    index.insert(4L, new float[] {10.0f, 10.0f});

    float[] query = {0.9f, 0.9f};
    List<SearchResult> results = index.searchKnn(query, 2);

    assertThat(results).hasSize(2);
    // Nearest must be point 2 (1.0, 1.0)
    assertThat(results.get(0).id()).isEqualTo(2L);
    // Second nearest must be point 1 (0.0, 0.0)
    assertThat(results.get(1).id()).isEqualTo(1L);
  }

  @Test
  @DisplayName("128D Ground Truth Oracle: Exact match with independent full-sort on 1,000 vectors")
  void test128DGroundTruthOracle() {
    int dimension = 128;
    int numVectors = 1000;
    int k = 10;
    Random rng = new Random(12345);

    FlatIndex index = new FlatIndex(dimension, DistanceMetric.EUCLIDEAN);
    ScalarEuclideanDistance referenceCalc = new ScalarEuclideanDistance();

    List<float[]> rawVectors = new ArrayList<>(numVectors);
    for (int i = 0; i < numVectors; i++) {
      float[] vec = new float[dimension];
      for (int d = 0; d < dimension; d++) {
        vec[d] = rng.nextFloat() * 2.0f - 1.0f;
      }
      rawVectors.add(vec);
      index.insert(i + 1L, vec);
    }

    // Query vector
    float[] query = new float[dimension];
    for (int d = 0; d < dimension; d++) {
      query[d] = rng.nextFloat() * 2.0f - 1.0f;
    }

    // 1. Query through FlatIndex
    List<SearchResult> indexResults = index.searchKnn(query, k);

    // 2. Query through independent ground truth list
    List<SearchResult> groundTruth = new ArrayList<>(numVectors);
    for (int i = 0; i < numVectors; i++) {
      float dist = referenceCalc.distance(rawVectors.get(i), query);
      groundTruth.add(new SearchResult(i + 1L, dist));
    }
    Collections.sort(groundTruth);
    List<SearchResult> expectedTopK = groundTruth.subList(0, k);

    // 3. Verify 100% exact equality on both ID and distance
    assertThat(indexResults).hasSize(k);
    for (int i = 0; i < k; i++) {
      assertThat(indexResults.get(i).id())
          .as("Rank %d ID must match Ground Truth", i)
          .isEqualTo(expectedTopK.get(i).id());

      assertThat(indexResults.get(i).distance())
          .as("Rank %d distance must match Ground Truth", i)
          .isCloseTo(expectedTopK.get(i).distance(), offset(1e-5f));
    }
  }

  @Test
  @DisplayName("Cosine: Auto-normalization finds most collinear vectors")
  void testCosineSearch() {
    FlatIndex index = new FlatIndex(2, DistanceMetric.COSINE);
    // Unnormalized inputs
    index.insert(10L, new float[] {1.0f, 0.0f});
    index.insert(20L, new float[] {3.0f, 4.0f});
    index.insert(30L, new float[] {0.0f, 5.0f});

    // Query collinear with (3, 4) -> (6, 8)
    float[] query = {6.0f, 8.0f};
    List<SearchResult> results = index.searchKnn(query, 1);

    assertThat(results).hasSize(1);
    assertThat(results.get(0).id()).isEqualTo(20L);
    assertThat(results.get(0).distance()).isCloseTo(0.0f, offset(1e-6f));
  }

  @Test
  @DisplayName("Edge Case: k > size returns all vectors sorted")
  void testKGreaterThanSize() {
    FlatIndex index = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    index.insert(1L, new float[] {1.0f, 1.0f});
    index.insert(2L, new float[] {2.0f, 2.0f});

    List<SearchResult> results = index.searchKnn(new float[] {0.0f, 0.0f}, 10);
    assertThat(results).hasSize(2);
    assertThat(results.get(0).id()).isEqualTo(1L);
    assertThat(results.get(1).id()).isEqualTo(2L);
  }

  @Test
  @DisplayName("Edge Case: Empty index returns empty list")
  void testEmptyIndex() {
    FlatIndex index = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    List<SearchResult> results = index.searchKnn(new float[] {1.0f, 1.0f}, 5);
    assertThat(results).isEmpty();
  }

  @Test
  @DisplayName("Edge Case: Invalid k or query dimension throws exception")
  void testInvalidParameters() {
    FlatIndex index = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    index.insert(1L, new float[] {1.0f, 1.0f});

    assertThatThrownBy(() -> index.searchKnn(new float[] {1.0f, 1.0f}, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("k must be positive");

    assertThatThrownBy(() -> index.searchKnn(new float[] {1.0f}, 5))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Invalid vector dimension");
  }

  @Test
  @DisplayName("Edge Case: Duplicate external ID is rejected")
  void testDuplicateExternalIdRejected() {
    FlatIndex index = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    index.insert(100L, new float[] {1.0f, 2.0f});

    assertThatThrownBy(() -> index.insert(100L, new float[] {3.0f, 4.0f}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Duplicate external ID: 100");
  }

  @Test
  @DisplayName("FlatIndex supports explicit SIMD and scalar constructors")
  void testFlatIndexSimdAndScalarModes() {
    FlatIndex defaultIndex = new FlatIndex(4, DistanceMetric.EUCLIDEAN);
    assertThat(defaultIndex.calculator()).isInstanceOf(ScalarEuclideanDistance.class);

    FlatIndex simdIndex = new FlatIndex(4, DistanceMetric.EUCLIDEAN, true);
    assertThat(simdIndex.calculator())
        .isInstanceOf(com.nanovector.core.distance.VectorEuclideanDistance.class);

    float[] v1 = {1.0f, 0.0f, 0.0f, 0.0f};
    float[] v2 = {0.0f, 1.0f, 0.0f, 0.0f};
    defaultIndex.insert(1L, v1);
    defaultIndex.insert(2L, v2);
    simdIndex.insert(1L, v1);
    simdIndex.insert(2L, v2);

    float[] query = {1.0f, 0.1f, 0.0f, 0.0f};
    List<SearchResult> defaultRes = defaultIndex.searchKnn(query, 2);
    List<SearchResult> simdRes = simdIndex.searchKnn(query, 2);

    assertThat(simdRes).hasSize(2);
    assertThat(simdRes.get(0).id()).isEqualTo(defaultRes.get(0).id()).isEqualTo(1L);
    assertThat(simdRes.get(0).distance()).isCloseTo(defaultRes.get(0).distance(), offset(1e-5f));
  }
}
