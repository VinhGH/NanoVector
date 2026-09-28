package com.nanovector.benchmark.quantization;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class FlatVsQuantizedFlatBenchmarkTest {

  @Test
  @DisplayName("Verify FlatVsQuantizedFlatBenchmark setup, candidate profiling, and percentiles")
  void testBenchmarkSetupAndProfiling() {
    FlatVsQuantizedFlatBenchmark benchmark = new FlatVsQuantizedFlatBenchmark();
    benchmark.initForTest(100);

    assertThat(benchmark.fp32Scalar().size()).isEqualTo(100);
    assertThat(benchmark.fp32Simd().size()).isEqualTo(100);
    assertThat(benchmark.sq8Scalar().size()).isEqualTo(100);
    assertThat(benchmark.sq8Simd().size()).isEqualTo(100);

    for (FlatVsQuantizedFlatBenchmark.Candidate candidate :
        FlatVsQuantizedFlatBenchmark.Candidate.values()) {
      FlatVsQuantizedFlatBenchmark.LatencyProfile prof =
          benchmark.profile(candidate, 10, 50, 100.0);

      assertThat(prof.candidate()).isEqualTo(candidate);
      assertThat(prof.vectorCount()).isEqualTo(100);
      assertThat(prof.throughputQps()).isGreaterThan(0.0);
      assertThat(prof.meanLatencyUs()).isGreaterThan(0.0);
      assertThat(prof.p50LatencyUs()).isGreaterThan(0.0);
      assertThat(prof.p95LatencyUs()).isGreaterThanOrEqualTo(prof.p50LatencyUs());
      assertThat(prof.p99LatencyUs()).isGreaterThanOrEqualTo(prof.p95LatencyUs());
      assertThat(prof.speedupVsFp32()).isGreaterThan(0.0);

      String row = prof.toMarkdownRow();
      assertThat(row).contains("100").contains(candidate.label());
    }
  }

  @Test
  @DisplayName("Verify memory comparison evaluation metrics and structural reduction")
  void testMemoryComparisonEvaluation() {
    FlatVsQuantizedFlatBenchmark.MemoryComparisonRow row =
        FlatVsQuantizedFlatBenchmark.evaluateMemory(100);

    assertThat(row.n()).isEqualTo(100);
    assertThat(row.fp32VectorStorageBytesPerVec()).isEqualTo(512.0);
    assertThat(row.sq8VectorStorageBytesPerVec()).isEqualTo(136.0);
    assertThat(row.vectorStorageReductionRatio())
        .isCloseTo(3.76, org.assertj.core.data.Offset.offset(0.01));
    assertThat(row.totalStructuralReductionRatio()).isGreaterThan(2.0);

    String md = row.toMarkdownRow();
    assertThat(md).contains("100").contains("512 B").contains("136 B");
  }

  @Test
  @DisplayName("Execute full Phase 6B Commit 4 Benchmark Sweep across 1K, 10K, 50K, 100K")
  void runFullBenchmarkSweep() {
    FlatVsQuantizedFlatBenchmark.main(new String[0]);
  }
}
