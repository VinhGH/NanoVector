package com.nanovector.benchmark.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.QuantizedHnswIndex;
import com.nanovector.core.storage.VectorStorage;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class ParetoFrontierRerankBenchmarkTest {

  @Test
  @DisplayName("Verify ParetoStepResult metrics and Two-Phase Re-ranking mathematical invariants")
  void testEvaluateStep() {
    int n = 200;
    int dim = 32;
    int k = 5;
    int efSearch = 25;
    int queries = 16;
    long dataSeed = 42L;
    long querySeed = 12345L;

    Random dataRng = new Random(dataSeed);
    float[][] dataset = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        dataset[i][d] = dataRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    Random queryRng = new Random(querySeed);
    float[][] testQueries = new float[queries][dim];
    for (int q = 0; q < queries; q++) {
      for (int d = 0; d < dim; d++) {
        testQueries[q][d] = queryRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    FlatIndex oracle = new FlatIndex(dim, DistanceMetric.EUCLIDEAN, n, true);
    HnswConfig config = HnswConfig.withSeed(dataSeed);
    HnswIndex fp32Index = new HnswIndex(dim, DistanceMetric.EUCLIDEAN, config, n, true);
    QuantizedHnswIndex pureSq8Index =
        new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, config, n, true);
    VectorStorage rawStorage = new VectorStorage(dim, n);

    for (int i = 0; i < n; i++) {
      oracle.insert(i, dataset[i]);
      fp32Index.insert(i, dataset[i]);
      pureSq8Index.insert(i, dataset[i]);
      rawStorage.insert(i, dataset[i]);
    }

    ParetoFrontierRerankBenchmark.ParetoStepResult step =
        ParetoFrontierRerankBenchmark.evaluateStep(
            n,
            dim,
            k,
            efSearch,
            queries,
            dataSeed,
            querySeed,
            oracle,
            fp32Index,
            pureSq8Index,
            rawStorage,
            testQueries);

    assertThat(step.vectorCount()).isEqualTo(n);
    assertThat(step.dimension()).isEqualTo(dim);
    assertThat(step.k()).isEqualTo(k);
    assertThat(step.efSearch()).isEqualTo(efSearch);

    // Recall bounds
    assertThat(step.fp32Recall()).isBetween(0.0, 1.0);
    assertThat(step.sq8Recall()).isBetween(0.0, 1.0);
    assertThat(step.candidateRecall()).isBetween(0.0, 1.0);
    assertThat(step.rerankedRecall()).isBetween(0.0, 1.0);

    // Fundamental mathematical invariant of Two-Phase Search:
    // Candidate Recall is the upper bound on Re-ranked Recall (you can't rank what wasn't
    // retrieved)
    assertThat(step.candidateRecall()).isGreaterThanOrEqualTo(step.rerankedRecall() - 1e-9);

    // Candidate list (efSearch >= k) contains at least as many GT hits as the top-k subset
    assertThat(step.candidateRecall()).isGreaterThanOrEqualTo(step.sq8Recall() - 1e-9);

    // Attribution identities
    assertThat(step.recallRecovered())
        .isCloseTo(step.rerankedRecall() - step.sq8Recall(), within(1e-9));
    assertThat(step.unrecoverableLoss()).isCloseTo(1.0 - step.candidateRecall(), within(1e-9));

    // Performance metrics
    assertThat(step.fp32Qps()).isGreaterThan(0.0);
    assertThat(step.sq8Qps()).isGreaterThan(0.0);
    assertThat(step.rerankedQps()).isGreaterThan(0.0);
    assertThat(step.fp32MeanLatencyUs()).isGreaterThan(0.0);
    assertThat(step.sq8MeanLatencyUs()).isGreaterThan(0.0);
    assertThat(step.rerankedMeanLatencyUs()).isGreaterThan(0.0);

    // Markdown formatting
    assertThat(step.toParetoMarkdownRow()).contains("|").contains("%");
    assertThat(step.toRerankDeepDiveMarkdownRow()).contains("|").contains("us");
    assertThat(step.toLatencyDistributionMarkdownRow()).contains("|").contains("/");
  }

  @Test
  @DisplayName("Verify runParetoSweep on small scale")
  void testRunParetoSweepSmall() {
    List<ParetoFrontierRerankBenchmark.ParetoStepResult> results =
        ParetoFrontierRerankBenchmark.runParetoSweep(100, 32, new int[] {10, 20});

    assertThat(results).hasSize(2);
    ParetoFrontierRerankBenchmark.printReportTables(results);
  }
}
