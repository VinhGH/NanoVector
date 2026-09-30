package com.nanovector.benchmark.allocation;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.core.model.SearchResult;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class OffHeapAllocationBenchmarkTest {

  @Test
  @DisplayName(
      "Verify OffHeapAllocationBenchmark setup, index sizes, and search execution across all variants")
  void testBenchmarkExecution() {
    OffHeapAllocationBenchmark bench = new OffHeapAllocationBenchmark();
    try {
      bench.initForTest(100);

      assertThat(bench.onHeapFp32Index().size()).isEqualTo(100);
      assertThat(bench.onHeapSq8Index().size()).isEqualTo(100);
      assertThat(bench.offHeapSq8Index().size()).isEqualTo(100);

      float[] query = new float[128];
      List<SearchResult> fp32Results = bench.onHeapFp32Index().searchKnn(query, 10);
      List<SearchResult> onHeapSq8Results = bench.onHeapSq8Index().searchKnn(query, 10);
      List<SearchResult> offHeapSq8Results = bench.offHeapSq8Index().searchKnn(query, 10);

      assertThat(fp32Results).hasSize(10);
      assertThat(onHeapSq8Results).hasSize(10);
      assertThat(offHeapSq8Results).hasSize(10);

      assertThat(fp32Results.get(0).distance()).isGreaterThanOrEqualTo(0.0f);
      assertThat(onHeapSq8Results.get(0).distance()).isGreaterThanOrEqualTo(0.0f);
      assertThat(offHeapSq8Results.get(0).distance()).isGreaterThanOrEqualTo(0.0f);
    } finally {
      bench.tearDown();
    }
  }
}
