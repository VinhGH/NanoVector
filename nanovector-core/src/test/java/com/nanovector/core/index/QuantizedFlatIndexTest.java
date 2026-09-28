package com.nanovector.core.index;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.model.SearchResult;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class QuantizedFlatIndexTest {

  @Test
  @DisplayName("Empty index search returns empty list")
  void testEmptySearch() {
    QuantizedFlatIndex index = new QuantizedFlatIndex(4);
    assertThat(index.size()).isEqualTo(0);
    assertThat(index.searchKnn(new float[] {1f, 2f, 3f, 4f}, 5)).isEmpty();
  }

  @Test
  @DisplayName("Basic insertion and Top-K retrieval matches expected ordering")
  void testBasicInsertionAndSearch() {
    int dim = 3;
    QuantizedFlatIndex index = new QuantizedFlatIndex(dim);

    index.insert(1L, new float[] {0.0f, 0.0f, 0.0f});
    index.insert(2L, new float[] {10.0f, 10.0f, 10.0f});
    index.insert(3L, new float[] {1.0f, 1.0f, 1.0f});

    assertThat(index.size()).isEqualTo(3);
    assertThat(index.dimension()).isEqualTo(dim);
    assertThat(index.metric()).isEqualTo(DistanceMetric.EUCLIDEAN);

    List<SearchResult> results = index.searchKnn(new float[] {0.1f, 0.1f, 0.1f}, 2);
    assertThat(results).hasSize(2);
    assertThat(results.get(0).id()).isEqualTo(1L);
    assertThat(results.get(1).id()).isEqualTo(3L);
    assertThat(results.get(0).distance()).isLessThan(results.get(1).distance());
  }

  @Test
  @DisplayName(
      "Zero-range constant vector search returns distance 0.0 to query matching same value")
  void testZeroRangeVectorSearch() {
    int dim = 16;
    QuantizedFlatIndex index = new QuantizedFlatIndex(dim);

    float[] constantVector = new float[dim];
    Arrays.fill(constantVector, 0.75f);
    index.insert(99L, constantVector);

    List<SearchResult> results = index.searchKnn(constantVector, 1);
    assertThat(results).hasSize(1);
    assertThat(results.get(0).id()).isEqualTo(99L);
    assertThat(results.get(0).distance()).isEqualTo(0.0f);
  }

  @Test
  @DisplayName(
      "Validation checks: k, dimensions, NaN, non-Euclidean metric, vectorData unsupported")
  void testValidationChecks() {
    QuantizedFlatIndex index = new QuantizedFlatIndex(4);
    index.insert(1L, new float[] {1f, 2f, 3f, 4f});

    assertThatThrownBy(() -> index.searchKnn(new float[] {1f, 2f, 3f, 4f}, 0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> index.searchKnn(new float[] {1f, 2f, 3f}, 1))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> index.searchKnn(new float[] {1f, Float.NaN, 3f, 4f}, 1))
        .isInstanceOf(IllegalArgumentException.class);

    assertThatThrownBy(() -> new QuantizedFlatIndex(4, DistanceMetric.COSINE, 10, true))
        .isInstanceOf(UnsupportedOperationException.class)
        .hasMessageContaining("EUCLIDEAN");

    assertThatThrownBy(index::vectorData)
        .isInstanceOf(UnsupportedOperationException.class)
        .hasMessageContaining("QuantizedVectorStorage");
  }

  @Test
  @DisplayName("Scalar and SIMD QuantizedFlatIndex produce identical Top-K nearest neighbors")
  void testScalarSimdSearchParity() {
    int dim = 64;
    int n = 100;
    Random rng = new Random(12345);

    QuantizedFlatIndex scalarIndex = new QuantizedFlatIndex(dim, false);
    QuantizedFlatIndex simdIndex = new QuantizedFlatIndex(dim, true);

    for (int i = 0; i < n; i++) {
      float[] vec = new float[dim];
      for (int d = 0; d < dim; d++) {
        vec[d] = rng.nextFloat(-5.0f, 5.0f);
      }
      scalarIndex.insert(i, vec);
      simdIndex.insert(i, vec);
    }

    float[] query = new float[dim];
    for (int d = 0; d < dim; d++) {
      query[d] = rng.nextFloat(-5.0f, 5.0f);
    }

    List<SearchResult> scalarRes = scalarIndex.searchKnn(query, 10);
    List<SearchResult> simdRes = simdIndex.searchKnn(query, 10);

    assertThat(scalarRes).hasSize(10);
    assertThat(simdRes).hasSize(10);

    for (int i = 0; i < 10; i++) {
      assertThat(simdRes.get(i).id()).isEqualTo(scalarRes.get(i).id());
      assertThat(simdRes.get(i).distance())
          .isCloseTo(scalarRes.get(i).distance(), org.assertj.core.data.Offset.offset(1e-3f));
    }
  }

  @Test
  @DisplayName(
      "Empirical Recall@10 of SQ8 flat baseline exceeds 95% against FP32 ground truth oracle")
  void testRecallAgainstGroundTruthOracle() {
    int dim = 128;
    int n = 1000;
    int numQueries = 50;
    int k = 10;
    Random rng = new Random(2026_09_28);

    FlatIndex fp32Oracle = new FlatIndex(dim, DistanceMetric.EUCLIDEAN, n);
    QuantizedFlatIndex sq8Index = new QuantizedFlatIndex(dim, n);

    for (int i = 0; i < n; i++) {
      float[] vec = new float[dim];
      for (int d = 0; d < dim; d++) {
        vec[d] = rng.nextFloat(-1.0f, 1.0f);
      }
      fp32Oracle.insert(i, vec);
      sq8Index.insert(i, vec);
    }

    double totalRecall = 0.0;
    for (int q = 0; q < numQueries; q++) {
      float[] query = new float[dim];
      for (int d = 0; d < dim; d++) {
        query[d] = rng.nextFloat(-1.0f, 1.0f);
      }

      List<SearchResult> exact = fp32Oracle.searchKnn(query, k);
      List<SearchResult> sq8 = sq8Index.searchKnn(query, k);

      Set<Long> exactIds = new HashSet<>();
      for (SearchResult r : exact) {
        exactIds.add(r.id());
      }

      int matches = 0;
      for (SearchResult r : sq8) {
        if (exactIds.contains(r.id())) {
          matches++;
        }
      }

      totalRecall += (double) matches / k;
    }

    double meanRecall = totalRecall / numQueries;
    System.out.printf(
        "SQ8 Flat Baseline Recall@%d over %d queries (N=%d, D=%d): %.2f%%\n",
        k, numQueries, n, dim, meanRecall * 100.0);

    // Theoretical and practical expectation for SQ8 on D=128 random vectors is > 95%
    assertThat(meanRecall).isGreaterThan(0.95);
  }
}
