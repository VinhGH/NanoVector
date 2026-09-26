package com.nanovector.benchmark.search;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.core.model.SearchResult;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class EfSearchRecallLatencyBenchmarkTest {

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  @DisplayName("Verify EfSearchRecallLatencyBenchmark setup and execution across SIMD modes")
  void testBenchmarkExecution(boolean useSimd) {
    EfSearchRecallLatencyBenchmark bench = new EfSearchRecallLatencyBenchmark();
    bench.initForTest(500, 50, useSimd);

    assertThat(bench.hnswIndex().size()).isEqualTo(500);
    assertThat(bench.getMeasuredRecall())
        .as("Recall@10 at efSearch=50 should exceed 85% on 500 vectors")
        .isGreaterThan(0.85);

    float[] query = new float[128];
    List<SearchResult> results = bench.hnswIndex().searchKnn(query, 10, 50);
    assertThat(results).hasSize(10);
    assertThat(results.get(0).distance()).isGreaterThanOrEqualTo(0.0f);
  }

  @Test
  @DisplayName("Verify recall monotonicity across efSearch parameter values")
  void testRecallMonotonicity() {
    EfSearchRecallLatencyBenchmark lowEf = new EfSearchRecallLatencyBenchmark();
    lowEf.initForTest(500, 10, true);

    EfSearchRecallLatencyBenchmark highEf = new EfSearchRecallLatencyBenchmark();
    highEf.initForTest(500, 100, true);

    assertThat(highEf.getMeasuredRecall())
        .as("Recall@10 at efSearch=100 should be greater than or equal to efSearch=10")
        .isGreaterThanOrEqualTo(lowEf.getMeasuredRecall());
  }
}
