package com.nanovector.benchmark.distance;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.nanovector.core.distance.DistanceMetric;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class ScalarVsSimdDistanceBenchmarkTest {

  private static final int[] DIMENSIONS = {32, 64, 128, 384, 768, 1536};

  @ParameterizedTest
  @EnumSource(DistanceMetric.class)
  @DisplayName("Verify semantic equivalence across all metrics and dimensions")
  void testBenchmarkEquivalenceAcrossDimensions(DistanceMetric metric) {
    for (int dim : DIMENSIONS) {
      ScalarVsSimdDistanceBenchmark benchmark = new ScalarVsSimdDistanceBenchmark();
      benchmark.initForTest(metric, dim);

      float scalarDist = benchmark.scalarDistance();
      float simdDist = benchmark.simdDistance();

      assertThat(Float.isFinite(scalarDist))
          .as("Scalar distance must be finite for %s D=%d", metric, dim)
          .isTrue();
      assertThat(Float.isFinite(simdDist))
          .as("SIMD distance must be finite for %s D=%d", metric, dim)
          .isTrue();

      float tolerance = (metric == DistanceMetric.COSINE) ? 1e-4f : 1e-3f;
      assertThat(simdDist)
          .as("SIMD and Scalar distance must match within tolerance for %s D=%d", metric, dim)
          .isCloseTo(scalarDist, within(tolerance));
    }
  }
}
