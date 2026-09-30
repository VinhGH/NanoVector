package com.nanovector.benchmark.search;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.benchmark.search.OnHeapVsOffHeapSearchBenchmark.BenchmarkSweepResult;
import com.nanovector.benchmark.search.OnHeapVsOffHeapSearchBenchmark.ConstructionResult;
import com.nanovector.benchmark.search.OnHeapVsOffHeapSearchBenchmark.SearchStepResult;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.QuantizedHnswIndex;
import com.nanovector.core.offheap.OffHeapQuantizedHnswIndex;
import com.nanovector.core.storage.VectorStorage;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class OnHeapVsOffHeapSearchBenchmarkTest {

  @Test
  @DisplayName("Verify evaluateConstruction produces consistent timing and throughput metrics")
  void testEvaluateConstruction() {
    int n = 150;
    int dim = 32;
    Random rng = new Random(42L);
    float[][] dataset = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        dataset[i][d] = rng.nextFloat() * 2.0f - 1.0f;
      }
    }

    HnswConfig config = HnswConfig.withSeed(42L);
    ConstructionResult result =
        OnHeapVsOffHeapSearchBenchmark.evaluateConstruction(n, dim, dataset, config);

    assertThat(result.vectorCount()).isEqualTo(n);
    assertThat(result.dimension()).isEqualTo(dim);
    assertThat(result.fp32BuildMs()).isGreaterThanOrEqualTo(0);
    assertThat(result.onHeapSq8BuildMs()).isGreaterThanOrEqualTo(0);
    assertThat(result.offHeapSq8BuildMs()).isGreaterThanOrEqualTo(0);
    assertThat(result.fp32VecPerSec()).isGreaterThan(0.0);
    assertThat(result.onHeapSq8VecPerSec()).isGreaterThan(0.0);
    assertThat(result.offHeapSq8VecPerSec()).isGreaterThan(0.0);
    assertThat(result.offHeapOverheadVsOnHeapSq8()).isGreaterThan(0.0);
    assertThat(result.offHeapOverheadVsFp32()).isGreaterThan(0.0);

    String row = result.toMarkdownRow();
    assertThat(row).contains("|").contains("x");
  }

  @Test
  @DisplayName("Verify evaluateSearchStep metrics, parity agreement, and latency distribution")
  void testEvaluateSearchStep() {
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
    QuantizedHnswIndex onHeapSq8Index =
        new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, config, n, true);
    OffHeapQuantizedHnswIndex offHeapSq8Index =
        new OffHeapQuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, config, n, true);
    VectorStorage rawStorage = new VectorStorage(dim, n);

    for (int i = 0; i < n; i++) {
      oracle.insert(i, dataset[i]);
      fp32Index.insert(i, dataset[i]);
      onHeapSq8Index.insert(i, dataset[i]);
      offHeapSq8Index.insert(i, dataset[i]);
      rawStorage.insert(i, dataset[i]);
    }

    try {
      SearchStepResult step =
          OnHeapVsOffHeapSearchBenchmark.evaluateSearchStep(
              n,
              dim,
              k,
              efSearch,
              queries,
              oracle,
              fp32Index,
              onHeapSq8Index,
              offHeapSq8Index,
              rawStorage,
              testQueries);

      assertThat(step.vectorCount()).isEqualTo(n);
      assertThat(step.dimension()).isEqualTo(dim);
      assertThat(step.k()).isEqualTo(k);
      assertThat(step.efSearch()).isEqualTo(efSearch);

      // Recall bounds
      assertThat(step.fp32Recall()).isBetween(0.0, 1.0);
      assertThat(step.onHeapSq8Recall()).isBetween(0.0, 1.0);
      assertThat(step.offHeapSq8Recall()).isBetween(0.0, 1.0);
      assertThat(step.rerankedRecall()).isBetween(0.0, 1.0);

      // Parity agreement rate should be high between On-Heap SQ8 and Off-Heap SQ8
      assertThat(step.parityAgreementRate()).isBetween(0.0, 1.0);
      assertThat(step.parityAgreementRate()).isGreaterThan(0.70);

      // Throughput & Speedup metrics
      assertThat(step.fp32Qps()).isGreaterThan(0.0);
      assertThat(step.onHeapSq8Qps()).isGreaterThan(0.0);
      assertThat(step.offHeapSq8Qps()).isGreaterThan(0.0);
      assertThat(step.rerankedQps()).isGreaterThan(0.0);
      assertThat(step.offHeapSpeedupVsOnHeapSq8()).isGreaterThan(0.0);

      // Latency metrics
      assertThat(step.fp32MeanUs()).isGreaterThan(0.0);
      assertThat(step.onHeapSq8MeanUs()).isGreaterThan(0.0);
      assertThat(step.offHeapSq8MeanUs()).isGreaterThan(0.0);
      assertThat(step.rerankedMeanUs()).isGreaterThan(0.0);

      // Markdown formatting
      assertThat(step.toSearchLatencyMarkdownRow()).contains("|").contains("/");
      assertThat(step.toAccuracyParityMarkdownRow()).contains("|").contains("%");
    } finally {
      offHeapSq8Index.close();
    }
  }

  @Test
  @DisplayName("Verify runBenchmarkSweep end-to-end execution on small scale")
  void testRunBenchmarkSweepSmall() {
    BenchmarkSweepResult result =
        OnHeapVsOffHeapSearchBenchmark.runBenchmarkSweep(100, 32, new int[] {10, 20});

    assertThat(result.construction()).isNotNull();
    assertThat(result.searchSteps()).hasSize(2);
    OnHeapVsOffHeapSearchBenchmark.printReportTables(result);
  }

  @Test
  @DisplayName("Verify JMH setup and search methods execute cleanly")
  void testJmhBenchmarkLifecycle() {
    OnHeapVsOffHeapSearchBenchmark bench = new OnHeapVsOffHeapSearchBenchmark();
    bench.initForTest(100);

    try {
      assertThat(bench.onHeapFp32Index()).isNotNull();
      assertThat(bench.onHeapSq8Index()).isNotNull();
      assertThat(bench.offHeapSq8Index()).isNotNull();

      bench.searchOnHeapFp32(null);
      bench.searchOnHeapSq8(null);
      bench.searchOffHeapSq8(null);
      bench.searchOffHeapSq8_Rerank(null);
    } finally {
      bench.tearDown();
    }
  }
}
