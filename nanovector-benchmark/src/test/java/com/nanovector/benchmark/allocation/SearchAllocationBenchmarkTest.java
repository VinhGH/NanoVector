package com.nanovector.benchmark.allocation;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.core.model.SearchResult;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class SearchAllocationBenchmarkTest {

  @Test
  @DisplayName("Verify SearchAllocationBenchmark setup and all benchmark methods execution")
  void testBenchmarkExecution() {
    SearchAllocationBenchmark bench = new SearchAllocationBenchmark();
    bench.initForTest(100);

    assertThat(bench.flatIndex().size()).isEqualTo(100);
    assertThat(bench.hnswIndex().size()).isEqualTo(100);

    float[] query = new float[128];
    List<SearchResult> flatResults = bench.flatIndex().searchKnn(query, 10);
    List<SearchResult> hnswResults10 = bench.hnswIndex().searchKnn(query, 10, 10);
    List<SearchResult> hnswResults50 = bench.hnswIndex().searchKnn(query, 10, 50);
    List<SearchResult> hnswResults100 = bench.hnswIndex().searchKnn(query, 10, 100);

    assertThat(flatResults).hasSize(10);
    assertThat(hnswResults10).hasSize(10);
    assertThat(hnswResults50).hasSize(10);
    assertThat(hnswResults100).hasSize(10);

    assertThat(flatResults.get(0).distance()).isGreaterThanOrEqualTo(0.0f);
    assertThat(hnswResults10.get(0).distance()).isGreaterThanOrEqualTo(0.0f);
  }
}
