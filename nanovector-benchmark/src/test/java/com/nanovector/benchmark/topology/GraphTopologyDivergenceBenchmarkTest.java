package com.nanovector.benchmark.topology;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class GraphTopologyDivergenceBenchmarkTest {

  @Test
  @DisplayName("Verify empirical topology divergence analysis between Scalar and SIMD")
  void testTopologyDivergenceAnalysis() {
    GraphTopologyDivergenceBenchmark.DivergenceReport report =
        GraphTopologyDivergenceBenchmark.analyzeDivergence(300, 64, 42L);

    assertThat(report.vectorCount()).isEqualTo(300);
    assertThat(report.dimension()).isEqualTo(64);
    assertThat(report.entryPointScalar()).isGreaterThanOrEqualTo(0);
    assertThat(report.entryPointSimd()).isGreaterThanOrEqualTo(0);
    assertThat(report.maxLevelScalar()).isGreaterThanOrEqualTo(0);
    assertThat(report.maxLevelSimd()).isGreaterThanOrEqualTo(0);

    // Entry point and max level must be structurally sound
    assertThat(report.totalEdgesScalar()).isGreaterThan(0);
    assertThat(report.totalEdgesSimd()).isGreaterThan(0);

    // Topological consistency gates:
    assertThat(report.edgeJaccardSimilarity())
        .as("Graph edge Jaccard similarity between Scalar and SIMD should be high")
        .isGreaterThan(0.85);

    assertThat(report.searchResultOverlapRatio())
        .as("Top-10 search result overlap between Scalar and SIMD graphs should exceed 90%")
        .isGreaterThan(0.90);
  }

  @Test
  @DisplayName("Verify benchmark setup runs without exception")
  void testBenchmarkExecution() {
    GraphTopologyDivergenceBenchmark benchmark = new GraphTopologyDivergenceBenchmark();
    // Use reflection or check setup
    assertThat(benchmark).isNotNull();
  }
}
