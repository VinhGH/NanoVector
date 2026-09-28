package com.nanovector.benchmark.memory;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.core.hnsw.HnswConfig;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class MemoryFootprintProfilerTest {

  @Test
  @DisplayName("Verify FlatIndex memory characterization metrics")
  void testProfileFlat() {
    int count = 200;
    int dim = 32;

    MemoryFootprintProfiler.MemoryReport report = MemoryFootprintProfiler.profileFlat(count, dim);

    assertThat(report.indexType()).isEqualTo("FlatIndex");
    assertThat(report.vectorCount()).isEqualTo(count);
    assertThat(report.dimension()).isEqualTo(dim);

    long expectedRawBytes = (long) count * dim * Float.BYTES;
    assertThat(report.rawVectorPayloadBytes()).isEqualTo(expectedRawBytes);
    assertThat(report.externalIdPayloadBytes()).isEqualTo((long) count * Long.BYTES);

    assertThat(report.graphStructuralBytes()).isEqualTo(0L);
    assertThat(report.graphOverheadPerVector()).isEqualTo(0.0);

    assertThat(report.totalStructuralBytes()).isGreaterThan(expectedRawBytes);
    assertThat(report.structuralBytesPerVector()).isGreaterThan(dim * Float.BYTES);
    assertThat(report.structuralAmplification()).isGreaterThan(1.0);

    String formatted = report.toFormattedString();
    assertThat(formatted)
        .contains("FlatIndex")
        .contains("Raw Vector Payload")
        .contains("Structural Amplification");
  }

  @Test
  @DisplayName("Verify HnswIndex memory characterization metrics and graph overhead")
  void testProfileHnsw() {
    int count = 200;
    int dim = 32;
    HnswConfig config = HnswConfig.withSeed(42L).withEfSearch(50);

    MemoryFootprintProfiler.MemoryReport flatReport =
        MemoryFootprintProfiler.profileFlat(count, dim);
    MemoryFootprintProfiler.MemoryReport hnswReport =
        MemoryFootprintProfiler.profileHnsw(count, dim, config);

    assertThat(hnswReport.indexType()).isEqualTo("HnswIndex");
    assertThat(hnswReport.graphStructuralBytes()).isGreaterThan(0L);
    assertThat(hnswReport.graphOverheadPerVector()).isGreaterThan(0.0);

    // HNSW must have higher structural footprint than Flat due to graph topology
    assertThat(hnswReport.totalStructuralBytes()).isGreaterThan(flatReport.totalStructuralBytes());
    assertThat(hnswReport.structuralAmplification())
        .isGreaterThan(flatReport.structuralAmplification());

    String formatted = hnswReport.toFormattedString();
    assertThat(formatted)
        .contains("HnswIndex")
        .contains("Graph Structural Estimate")
        .contains("Graph Overhead / Vector")
        .contains("Structural Amplification");
  }

  @Test
  @DisplayName("Verify QuantizedFlatIndex memory characterization and structural reduction")
  void testProfileQuantizedFlat() {
    int count = 200;
    int dim = 128;

    MemoryFootprintProfiler.MemoryReport flatReport =
        MemoryFootprintProfiler.profileFlat(count, dim);
    MemoryFootprintProfiler.MemoryReport sq8Report =
        MemoryFootprintProfiler.profileQuantizedFlat(count, dim);

    assertThat(sq8Report.indexType()).isEqualTo("QuantizedFlatIndex");
    assertThat(sq8Report.vectorCount()).isEqualTo(count);
    assertThat(sq8Report.dimension()).isEqualTo(dim);

    // Structural storage for SQ8 must be significantly smaller than FP32 Flat
    assertThat(sq8Report.storageStructuralBytes()).isLessThan(flatReport.storageStructuralBytes());
    assertThat(sq8Report.totalStructuralBytes()).isLessThan(flatReport.totalStructuralBytes());

    // Vector coordinate structural storage per vector should be ~136 bytes for D=128
    double structuralRatio =
        (double) flatReport.storageStructuralBytes() / sq8Report.storageStructuralBytes();
    assertThat(structuralRatio).isGreaterThan(2.0);

    String formatted = sq8Report.toFormattedString();
    assertThat(formatted)
        .contains("QuantizedFlatIndex")
        .contains("Raw Vector Payload")
        .contains("Storage Structural Estimate");
  }
}
