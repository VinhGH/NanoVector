package com.nanovector.benchmark.scale;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.core.model.SearchResult;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class ScaleLatencySearchBenchmarkTest {

  @Test
  @DisplayName("Verify ScaleLatencySearchBenchmark setup, query correctness, and recall estimation")
  void testBenchmarkSetupAndSearch() {
    ScaleLatencySearchBenchmark benchmark = new ScaleLatencySearchBenchmark();
    benchmark.initForTest(100);

    assertThat(benchmark.flatIndex().size()).isEqualTo(100);
    assertThat(benchmark.hnswIndex().size()).isEqualTo(100);
    assertThat(benchmark.workingSetMiB()).isGreaterThan(0.0);
    assertThat(benchmark.measuredRecall()).isBetween(0.80, 1.0);

    float[] query = new float[128];
    List<SearchResult> flatResults = benchmark.flatIndex().searchKnn(query, 10);
    List<SearchResult> hnswResults = benchmark.hnswIndex().searchKnn(query, 10, 50);

    assertThat(flatResults).hasSize(10);
    assertThat(hnswResults).hasSize(10);
    assertThat(flatResults.get(0).distance()).isGreaterThanOrEqualTo(0.0f);
    assertThat(hnswResults.get(0).distance()).isGreaterThanOrEqualTo(0.0f);
  }

  @Test
  @DisplayName("Verify latency distribution profiling across percentiles (P50, P90, P99)")
  void testProfileIndexPercentiles() {
    ScaleLatencySearchBenchmark benchmark = new ScaleLatencySearchBenchmark();
    benchmark.initForTest(100);

    ScaleLatencySearchBenchmark.LatencyProfile flatProf = benchmark.profileIndex(true, 10, 100);
    ScaleLatencySearchBenchmark.LatencyProfile hnswProf = benchmark.profileIndex(false, 10, 100);

    assertThat(flatProf.indexType()).isEqualTo("FlatIndex");
    assertThat(flatProf.vectorCount()).isEqualTo(100);
    assertThat(flatProf.throughputOpsPerSec()).isGreaterThan(0.0);
    assertThat(flatProf.meanLatencyUs()).isGreaterThan(0.0);
    assertThat(flatProf.p50LatencyUs()).isGreaterThan(0.0);
    assertThat(flatProf.p90LatencyUs()).isGreaterThanOrEqualTo(flatProf.p50LatencyUs());
    assertThat(flatProf.p99LatencyUs()).isGreaterThanOrEqualTo(flatProf.p90LatencyUs());
    assertThat(flatProf.maxLatencyUs()).isGreaterThanOrEqualTo(flatProf.p99LatencyUs());

    assertThat(hnswProf.indexType()).isEqualTo("HnswIndex");
    assertThat(hnswProf.vectorCount()).isEqualTo(100);
    assertThat(hnswProf.recallAt10()).isBetween(0.80, 1.0);
    assertThat(hnswProf.p50LatencyUs()).isGreaterThan(0.0);
    assertThat(hnswProf.p90LatencyUs()).isGreaterThanOrEqualTo(hnswProf.p50LatencyUs());
    assertThat(hnswProf.p99LatencyUs()).isGreaterThanOrEqualTo(hnswProf.p90LatencyUs());

    String row = hnswProf.toMarkdownRow(flatProf.meanLatencyUs());
    assertThat(row).contains("HnswIndex").contains("100");
  }
}
