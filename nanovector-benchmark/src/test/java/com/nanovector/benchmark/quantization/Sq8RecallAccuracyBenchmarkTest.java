package com.nanovector.benchmark.quantization;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class Sq8RecallAccuracyBenchmarkTest {

  @Test
  @DisplayName("Verify Sq8RecallAccuracyBenchmark evaluation execution and metrics integrity")
  void testBenchmarkEvaluation() {
    int n = 200;
    int dim = 32;
    int k = 5;
    int queries = 16;

    Sq8RecallAccuracyBenchmark.RecallSweepRow result =
        Sq8RecallAccuracyBenchmark.evaluate(n, dim, k, queries, 42L, 12345L);

    assertThat(result.n()).isEqualTo(n);
    assertThat(result.dimension()).isEqualTo(dim);
    assertThat(result.k()).isEqualTo(k);
    assertThat(result.numQueries()).isEqualTo(queries);

    assertThat(result.fp32Recall()).isEqualTo(1.0);
    assertThat(result.sq8ScalarRecall()).isGreaterThan(0.80);
    assertThat(result.sq8SimdRecall()).isGreaterThan(0.80);
    assertThat(result.scalarSimdAgreement()).isGreaterThan(0.95);
    assertThat(result.recallLoss()).isBetween(0.0, 0.20);

    String row = result.toMarkdownRow();
    assertThat(row).contains("200").contains("100.00%");
  }

  @Test
  @DisplayName("Scalar and SIMD Recall results are numerically aligned")
  void testScalarSimdParity() {
    int n = 300;
    int dim = 64;
    int k = 10;
    int queries = 20;

    Sq8RecallAccuracyBenchmark.RecallSweepRow result =
        Sq8RecallAccuracyBenchmark.evaluate(n, dim, k, queries, 100L, 200L);

    // Scalar and SIMD should not diverge in any meaningful way
    assertThat(Math.abs(result.sq8ScalarRecall() - result.sq8SimdRecall()))
        .isLessThanOrEqualTo(0.05);
    assertThat(result.scalarSimdAgreement()).isGreaterThanOrEqualTo(0.98);
  }

  @Test
  @DisplayName("Execute full SQ8 Recall Accuracy Benchmark sweep across 1K, 10K, 50K, 100K")
  void runFullSweep() {
    Sq8RecallAccuracyBenchmark.main(new String[0]);
  }
}
