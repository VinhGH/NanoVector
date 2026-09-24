package com.nanovector.core.index;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.model.SearchResult;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Empirical Recall@10 verification of {@link HnswIndex} against the {@link FlatIndex} Ground Truth
 * Oracle.
 *
 * <p>Evaluates recall on a 1,000-vector dataset of 128-dimensional vectors with 50 test queries.
 * Measures recall across different {@code efSearch} values: 10, 20, 50, 100.
 */
class HnswRecallTest {

  private static final int DIMENSION = 128;
  private static final int NUM_VECTORS = 1000;
  private static final int NUM_QUERIES = 50;
  private static final int K = 10;
  private static final long SEED = 42L;

  private static float[] randomVector(Random rng, int dim) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = rng.nextFloat() * 2f - 1f;
    }
    return v;
  }

  @Test
  @DisplayName("Measure Recall@10 against FlatIndex Ground Truth across efSearch values")
  void testRecallAtKAgainstFlatIndex() {
    Random dataRng = new Random(SEED);

    // Generate dataset
    List<float[]> dataset = new ArrayList<>(NUM_VECTORS);
    for (int i = 0; i < NUM_VECTORS; i++) {
      dataset.add(randomVector(dataRng, DIMENSION));
    }

    // Generate queries
    Random queryRng = new Random(SEED + 1);
    List<float[]> queries = new ArrayList<>(NUM_QUERIES);
    for (int i = 0; i < NUM_QUERIES; i++) {
      queries.add(randomVector(queryRng, DIMENSION));
    }

    // Build FlatIndex (Ground Truth Oracle)
    FlatIndex flatIndex = new FlatIndex(DIMENSION, DistanceMetric.EUCLIDEAN);
    for (int i = 0; i < NUM_VECTORS; i++) {
      flatIndex.insert(i, dataset.get(i));
    }

    // Build HnswIndex
    HnswConfig config = HnswConfig.defaultConfig().withSeed(SEED);
    HnswIndex hnswIndex = new HnswIndex(DIMENSION, DistanceMetric.EUCLIDEAN, config);
    for (int i = 0; i < NUM_VECTORS; i++) {
      hnswIndex.insert(i, dataset.get(i));
    }

    // Compute Ground Truth results for all queries
    List<Set<Long>> groundTruthSets = new ArrayList<>(NUM_QUERIES);
    for (float[] query : queries) {
      List<SearchResult> gtResults = flatIndex.searchKnn(query, K);
      Set<Long> gtIds = new HashSet<>();
      for (SearchResult r : gtResults) {
        gtIds.add(r.id());
      }
      groundTruthSets.add(gtIds);
    }

    // Measure Recall@10 at various efSearch values
    int[] efSearchValues = {10, 20, 50, 100};
    double[] recalls = new double[efSearchValues.length];

    System.out.println(
        "=== HNSW Recall@10 vs FlatIndex Oracle (N=1000, D=128, Q=50, M=16, efConstruction=200) ===");

    for (int e = 0; e < efSearchValues.length; e++) {
      int ef = efSearchValues[e];
      int totalHits = 0;

      for (int q = 0; q < NUM_QUERIES; q++) {
        float[] query = queries.get(q);
        Set<Long> gtIds = groundTruthSets.get(q);

        List<SearchResult> hnswResults = hnswIndex.searchKnn(query, K, ef);
        for (SearchResult r : hnswResults) {
          if (gtIds.contains(r.id())) {
            totalHits++;
          }
        }
      }

      double recall = (double) totalHits / (NUM_QUERIES * K);
      recalls[e] = recall;
      System.out.printf(
          "  efSearch = %3d -> Recall@10 = %.4f (%.2f%%)%n", ef, recall, recall * 100.0);
    }

    // Quality gates:
    // 1. Monotonicity: Recall must not decrease significantly as efSearch increases
    assertThat(recalls[3])
        .as("Recall at efSearch=100 should be higher than at efSearch=10")
        .isGreaterThanOrEqualTo(recalls[0]);

    // 2. High recall at efSearch=100 (HNSW standard should achieve > 80% on random uniform 128D)
    assertThat(recalls[3])
        .as("Recall@10 at efSearch=100 should achieve at least 80%")
        .isGreaterThan(0.80);
  }
}
