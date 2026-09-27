package com.nanovector.benchmark.scale;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.core.index.HnswIndex;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class ScaleConstructionBenchmarkTest {

  @Test
  @DisplayName("Verify ScaleConstructionBenchmark setup and index build execution")
  void testBenchmarkExecution() {
    ScaleConstructionBenchmark benchmark = new ScaleConstructionBenchmark();
    benchmark.initForTest(50);

    assertThat(benchmark.dataset()).hasDimensions(50, 128);

    HnswIndex index = benchmark.buildHnswIndex();
    assertThat(index.size()).isEqualTo(50);
    assertThat(index.graph().size()).isEqualTo(50);
  }

  @Test
  @DisplayName("Verify HNSW construction profiling, layer degree distribution, and invariants")
  void testProfileConstruction() {
    int count = 100;
    int dim = 32;

    ScaleConstructionBenchmark.ConstructionReport report =
        ScaleConstructionBenchmark.profileConstruction(count, dim);

    assertThat(report.vectorCount()).isEqualTo(count);
    assertThat(report.dimension()).isEqualTo(dim);
    assertThat(report.rawPayloadMiB()).isGreaterThan(0.0);
    assertThat(report.buildTimeMs()).isGreaterThanOrEqualTo(0L);
    assertThat(report.throughputVecPerSec()).isGreaterThan(0.0);
    assertThat(report.meanLatencyUsPerVec()).isGreaterThan(0.0);
    assertThat(report.maxLevel()).isGreaterThanOrEqualTo(0);
    assertThat(report.isolatedNodesLayer0()).isEqualTo(0);
    assertThat(report.totalEdgesAllLayers()).isGreaterThan(0L);

    assertThat(report.layerStats()).isNotEmpty();

    // Verify Layer 0 invariants
    ScaleConstructionBenchmark.LayerStats layer0 = report.layerStats().get(0);
    assertThat(layer0.layer()).isEqualTo(0);
    assertThat(layer0.nodeCount()).isEqualTo(count);
    assertThat(layer0.fractionOfTotal()).isEqualTo(1.0);
    assertThat(layer0.minDegree()).isGreaterThan(0);
    assertThat(layer0.maxDegree()).isLessThanOrEqualTo(layer0.maxAllowedDegree());
    assertThat(layer0.invariantPass()).isTrue();

    // Verify higher layer invariants
    for (int l = 1; l < report.layerStats().size(); l++) {
      ScaleConstructionBenchmark.LayerStats ls = report.layerStats().get(l);
      assertThat(ls.layer()).isEqualTo(l);
      assertThat(ls.nodeCount()).isLessThanOrEqualTo(report.layerStats().get(l - 1).nodeCount());
      assertThat(ls.maxDegree()).isLessThanOrEqualTo(ls.maxAllowedDegree());
      assertThat(ls.invariantPass()).isTrue();
    }

    String summaryRow = report.toSummaryMarkdownRow();
    assertThat(summaryRow).contains("100");

    String layerRow = layer0.toMarkdownRow();
    assertThat(layerRow).contains("Layer  0");
  }
}
