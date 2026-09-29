package com.nanovector.benchmark.topology;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class HnswConstructionStrategyBenchmarkTest {

  @Test
  @DisplayName(
      "Verify HnswConstructionStrategyBenchmark evaluation execution and metrics integrity")
  void testBenchmarkEvaluationFast() {
    int n = 300;
    int dim = 32;
    int k = 10;
    int efSearch = 30;
    int queries = 16;

    HnswConstructionStrategyBenchmark.StrategyEvaluationResult r =
        HnswConstructionStrategyBenchmark.evaluate(n, dim, k, efSearch, queries, 42L, 12345L);

    assertThat(r.vectorCount()).isEqualTo(n);
    assertThat(r.dimension()).isEqualTo(dim);
    assertThat(r.k()).isEqualTo(k);
    assertThat(r.efSearch()).isEqualTo(efSearch);
    assertThat(r.numQueries()).isEqualTo(queries);

    // Build metrics
    assertThat(r.fp32BuildTimeMs()).isGreaterThanOrEqualTo(0L);
    assertThat(r.pureSq8BuildTimeMs()).isGreaterThanOrEqualTo(0L);
    assertThat(r.fp32BuildThroughput()).isGreaterThan(0.0);
    assertThat(r.pureSq8BuildThroughput()).isGreaterThan(0.0);
    assertThat(r.buildTimeRatio()).isGreaterThan(0.0);

    // Topology metrics and invariants
    assertThat(r.layer0EdgeJaccard()).isBetween(0.0, 1.0);
    assertThat(r.fp32TotalEdges()).isGreaterThan(0L);
    assertThat(r.pureSq8TotalEdges()).isGreaterThan(0L);
    assertThat(r.pureSq8MaxDegreeL0()).isLessThanOrEqualTo(32);
    assertThat(r.pureSq8IsolatedL0()).isEqualTo(0);
    assertThat(r.pureSq8ComponentsL0()).isEqualTo(1);
    assertThat(r.pureSq8TopologyHealthy()).isTrue();

    // Recall metrics & scientific attribution identity
    assertThat(r.fp32Recall()).isGreaterThan(0.70);
    assertThat(r.hybridSq8Recall()).isGreaterThan(0.70);
    assertThat(r.pureSq8Recall()).isGreaterThan(0.70);
    assertThat(r.hybridPureAgreement()).isGreaterThan(0.70);

    double sumAttribution = r.quantizationDistanceLoss() + r.topologyDivergenceLoss();
    assertThat(r.totalRecallLoss()).isCloseTo(sumAttribution, within(1e-6));

    // Search throughput & speedup
    assertThat(r.fp32Qps()).isGreaterThan(0.0);
    assertThat(r.pureSq8Qps()).isGreaterThan(0.0);
    assertThat(r.searchSpeedup()).isGreaterThan(0.0);

    // Markdown row formatting
    assertThat(r.toBuildMarkdownRow()).contains("|").contains("ms");
    assertThat(r.toTopologyMarkdownRow()).contains("|").contains("true");
    assertThat(r.toRecallMarkdownRow()).contains("|").contains("%");
    assertThat(r.toSearchMarkdownRow()).contains("|").contains("x");
  }

  @Test
  @DisplayName("Verify sweep execution and table reporting on small scales")
  void testSweepExecutionSmall() {
    int[] testScales = {100, 200};
    List<HnswConstructionStrategyBenchmark.StrategyEvaluationResult> results =
        HnswConstructionStrategyBenchmark.runSweep(testScales);

    assertThat(results).hasSize(2);
    for (HnswConstructionStrategyBenchmark.StrategyEvaluationResult r : results) {
      assertThat(r.pureSq8TopologyHealthy()).isTrue();
      assertThat(r.pureSq8ComponentsL0()).isEqualTo(1);
      assertThat(r.pureSq8IsolatedL0()).isEqualTo(0);
    }

    // Verify printReportTables execution
    HnswConstructionStrategyBenchmark.printReportTables(results);
  }
}
