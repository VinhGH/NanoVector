package com.nanovector.benchmark.search;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.core.model.SearchResult;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class FlatVsHnswSearchBenchmarkTest {

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  @DisplayName("Verify search benchmark setup and execution for scalar and SIMD")
  void testSearchBenchmarkExecution(boolean useSimd) {
    FlatVsHnswSearchBenchmark benchmark = new FlatVsHnswSearchBenchmark();
    benchmark.initForTest(100, useSimd);

    assertThat(benchmark.flatIndex().size()).isEqualTo(100);
    assertThat(benchmark.hnswIndex().size()).isEqualTo(100);

    float[] query = new float[128];
    List<SearchResult> flatResults = benchmark.flatIndex().searchKnn(query, 10);
    List<SearchResult> hnswResults = benchmark.hnswIndex().searchKnn(query, 10);

    assertThat(flatResults).hasSize(10);
    assertThat(hnswResults).hasSize(10);

    assertThat(flatResults.get(0).distance()).isGreaterThanOrEqualTo(0.0f);
    assertThat(hnswResults.get(0).distance()).isGreaterThanOrEqualTo(0.0f);
  }
}
