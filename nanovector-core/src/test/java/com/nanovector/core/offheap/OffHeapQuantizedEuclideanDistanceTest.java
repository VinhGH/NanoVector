package com.nanovector.core.offheap;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.quantization.AsymmetricSq8Quantizer;
import com.nanovector.core.quantization.QuantizedVector;
import com.nanovector.core.quantization.ScalarQuantizedEuclideanDistance;
import com.nanovector.core.quantization.VectorQuantizedEuclideanDistance;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class OffHeapQuantizedEuclideanDistanceTest {

  private static final float RELATIVE_TOLERANCE = 1e-4f;
  private static final float ABSOLUTE_TOLERANCE = 1e-5f;

  @ParameterizedTest
  @ValueSource(ints = {5, 16, 32, 64, 127, 128, 256, 384, 512, 768, 1536})
  @DisplayName(
      "Should maintain numerical equivalence between on-heap and off-heap distance kernels across dimensions")
  void shouldMaintainNumericalEquivalenceAcrossDimensions(int dim) {
    Random random = new Random(42L + dim);
    float[] v = new float[dim];
    float[] q = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = random.nextFloat() * 10.0f - 5.0f;
      q[i] = random.nextFloat() * 10.0f - 5.0f;
    }

    QuantizedVector qv = AsymmetricSq8Quantizer.INSTANCE.quantize(v);
    byte[] onHeapBuffer = qv.data();
    float min = qv.min();
    float scale = qv.scale();

    // On-heap kernels
    ScalarQuantizedEuclideanDistance onHeapScalar = new ScalarQuantizedEuclideanDistance();
    VectorQuantizedEuclideanDistance onHeapSimd = new VectorQuantizedEuclideanDistance();
    float expectedScalar = onHeapScalar.distance(onHeapBuffer, 0, min, scale, q);
    float expectedSimd = onHeapSimd.distance(onHeapBuffer, 0, min, scale, q);

    // Off-heap kernels
    OffHeapScalarQuantizedEuclideanDistance offHeapScalar =
        new OffHeapScalarQuantizedEuclideanDistance();
    OffHeapVectorQuantizedEuclideanDistance offHeapSimd =
        new OffHeapVectorQuantizedEuclideanDistance();

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = arena.allocate(dim + 16, 8);
      long offset = 8L; // Test non-zero offset
      MemorySegment.copy(onHeapBuffer, 0, segment, ValueLayout.JAVA_BYTE, offset, dim);

      float actualScalar = offHeapScalar.distance(segment, offset, min, scale, q);
      float actualSimd = offHeapSimd.distance(segment, offset, min, scale, q);

      // Verify off-heap scalar equals on-heap scalar exactly
      assertThat(actualScalar)
          .as("Off-heap scalar vs on-heap scalar at dim %d", dim)
          .isEqualTo(expectedScalar);

      // Verify off-heap SIMD matches on-heap SIMD within tolerance
      assertClose(actualSimd, expectedSimd, "Off-heap SIMD vs on-heap SIMD at dim " + dim);

      // Verify off-heap SIMD matches off-heap scalar within tolerance
      assertClose(actualSimd, actualScalar, "Off-heap SIMD vs off-heap scalar at dim " + dim);
    }
  }

  @Test
  @DisplayName("Should handle scale == 0.0f (uniform constant vector) correctly")
  void shouldHandleZeroScale() {
    int dim = 128;
    float[] v = new float[dim];
    float[] q = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = 3.5f; // Constant vector -> min=3.5, scale=0.0
      q[i] = 1.0f;
    }

    QuantizedVector qv = AsymmetricSq8Quantizer.INSTANCE.quantize(v);
    assertThat(qv.scale()).isZero();

    OffHeapScalarQuantizedEuclideanDistance scalar = new OffHeapScalarQuantizedEuclideanDistance();
    OffHeapVectorQuantizedEuclideanDistance simd = new OffHeapVectorQuantizedEuclideanDistance();

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = arena.allocate(dim, 8);
      MemorySegment.copy(qv.data(), 0, segment, ValueLayout.JAVA_BYTE, 0L, dim);

      float distScalar = scalar.distance(segment, 0L, qv.min(), qv.scale(), q);
      float distSimd = simd.distance(segment, 0L, qv.min(), qv.scale(), q);

      float expected = (1.0f - 3.5f) * (1.0f - 3.5f) * dim; // (-2.5)^2 * 128 = 6.25 * 128 = 800
      assertThat(distScalar).isEqualTo(expected);
      assertClose(distSimd, expected, "SIMD zero scale");
    }
  }

  @Test
  @DisplayName("Should throw IndexOutOfBoundsException when segment bounds are exceeded")
  void shouldThrowOnOutOfBounds() {
    int dim = 64;
    OffHeapScalarQuantizedEuclideanDistance dist = new OffHeapScalarQuantizedEuclideanDistance();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = arena.allocate(32, 8);
      float[] query = new float[dim];

      assertThatThrownBy(() -> dist.distance(segment, 0L, 0f, 1f, query))
          .isInstanceOf(IndexOutOfBoundsException.class);

      assertThatThrownBy(() -> dist.distance(segment, -1L, 0f, 1f, query))
          .isInstanceOf(IndexOutOfBoundsException.class);
    }
  }

  private static void assertClose(float actual, float expected, String context) {
    float diff = Math.abs(actual - expected);
    float max = Math.max(Math.abs(actual), Math.abs(expected));
    if (max > 1e-4f) {
      float relError = diff / max;
      assertThat(relError)
          .as(context + " relative error (actual=" + actual + ", expected=" + expected + ")")
          .isLessThanOrEqualTo(RELATIVE_TOLERANCE);
    } else {
      assertThat(diff)
          .as(context + " absolute error (actual=" + actual + ", expected=" + expected + ")")
          .isLessThanOrEqualTo(ABSOLUTE_TOLERANCE);
    }
  }
}
