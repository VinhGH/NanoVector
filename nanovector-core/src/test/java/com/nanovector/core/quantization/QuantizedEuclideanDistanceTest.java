package com.nanovector.core.quantization;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.nanovector.core.distance.ScalarEuclideanDistance;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class QuantizedEuclideanDistanceTest {

  private final QuantizedEuclideanDistance scalarDist = new ScalarQuantizedEuclideanDistance();
  private final QuantizedEuclideanDistance simdDist = new VectorQuantizedEuclideanDistance();
  private final ScalarQuantizer quantizer = AsymmetricSq8Quantizer.INSTANCE;

  @Test
  @DisplayName("Zero-range constant vector distance to self is exactly 0.0 without NaN/Inf")
  void testZeroRangeVectorDistanceToSelfIsZero() {
    int dim = 128;
    float[] v = new float[dim];
    Arrays.fill(v, 0.75f);

    QuantizedVector qv = quantizer.quantize(v);
    assertThat(qv.scale()).isEqualTo(0.0f);
    assertThat(qv.min()).isEqualTo(0.75f);

    float dScalar = scalarDist.distance(qv.data(), 0, qv.min(), qv.scale(), v);
    float dSimd = simdDist.distance(qv.data(), 0, qv.min(), qv.scale(), v);

    assertThat(dScalar).isEqualTo(0.0f);
    assertThat(dSimd).isEqualTo(0.0f);
    assertThat(Float.isFinite(dScalar)).isTrue();
    assertThat(Float.isFinite(dSimd)).isTrue();
  }

  @Test
  @DisplayName("Zero-range vector distance to non-zero query is mathematically exact")
  void testZeroRangeVectorDistanceToDifferentQuery() {
    int dim = 128;
    float[] v = new float[dim];
    Arrays.fill(v, 0.75f);

    float[] query = new float[dim];
    Arrays.fill(query, 1.0f);

    QuantizedVector qv = quantizer.quantize(v);
    float expected = dim * (1.0f - 0.75f) * (1.0f - 0.75f); // 128 * 0.0625 = 8.0f

    float dScalar = scalarDist.distance(qv.data(), 0, qv.min(), qv.scale(), query);
    float dSimd = simdDist.distance(qv.data(), 0, qv.min(), qv.scale(), query);

    assertThat(dScalar).isCloseTo(expected, within(1e-5f));
    assertThat(dSimd).isCloseTo(expected, within(1e-5f));
  }

  @Test
  @DisplayName(
      "ADC distance matches full dequantization + Euclidean distance within float precision")
  void testGroundTruthFidelity() {
    int dim = 128;
    Random rng = new Random(42);

    float[] rawVector = new float[dim];
    float[] query = new float[dim];
    for (int i = 0; i < dim; i++) {
      rawVector[i] = rng.nextFloat(-10.0f, 10.0f);
      query[i] = rng.nextFloat(-10.0f, 10.0f);
    }

    QuantizedVector qv = quantizer.quantize(rawVector);
    float[] reconstructed = quantizer.dequantize(qv);

    // Baseline Euclidean distance on explicitly reconstructed float vector
    ScalarEuclideanDistance exactCalc = new ScalarEuclideanDistance();
    float exactDist = exactCalc.distance(query, reconstructed);

    float adcScalar = scalarDist.distance(qv, query);
    float adcSimd = simdDist.distance(qv, query);

    // ADC on-the-fly reconstruction must match explicit dequantization
    assertThat(adcScalar).isCloseTo(exactDist, within(1e-4f));
    assertThat(adcSimd).isCloseTo(exactDist, within(1e-4f));
  }

  @ParameterizedTest
  @ValueSource(
      ints = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 64, 127, 128, 129, 256, 384, 512, 1536})
  @DisplayName(
      "Scalar ADC and SIMD ADC produce numerically equivalent results across all dimensions")
  void testScalarSimdEquivalenceAcrossDimensions(int dim) {
    Random rng = new Random(100 + dim);

    float[] rawVector = new float[dim];
    float[] query = new float[dim];
    for (int i = 0; i < dim; i++) {
      rawVector[i] = rng.nextFloat(-100.0f, 100.0f);
      query[i] = rng.nextFloat(-100.0f, 100.0f);
    }

    QuantizedVector qv = quantizer.quantize(rawVector);

    float dScalar = scalarDist.distance(qv.data(), 0, qv.min(), qv.scale(), query);
    float dSimd = simdDist.distance(qv.data(), 0, qv.min(), qv.scale(), query);

    assertFloatClose(dSimd, dScalar, "ADC mismatch for dimension " + dim);
  }

  private static void assertFloatClose(float actual, float expected, String description) {
    float tolerance = Math.max(1e-4f, Math.abs(expected) * 1e-4f);
    assertThat(actual).as(description).isCloseTo(expected, within(tolerance));
  }

  @Test
  @DisplayName("Factory methods create valid instances according to useSimd parameter")
  void testFactoryMethods() {
    QuantizedEuclideanDistance s = QuantizedEuclideanDistance.create(false);
    assertThat(s).isInstanceOf(ScalarQuantizedEuclideanDistance.class);

    QuantizedEuclideanDistance v = QuantizedEuclideanDistance.create(true);
    assertThat(v).isInstanceOf(VectorQuantizedEuclideanDistance.class);

    QuantizedEuclideanDistance def = QuantizedEuclideanDistance.create();
    assertThat(def).isInstanceOf(VectorQuantizedEuclideanDistance.class);
  }

  @Test
  @DisplayName("Out of bounds and null checks are strictly enforced")
  void testBoundsAndNull() {
    byte[] buf = new byte[10];
    float[] q = new float[10];

    assertThatThrownBy(() -> scalarDist.distance(null, 0, 0f, 1f, q))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> scalarDist.distance(buf, 0, 0f, 1f, null))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> scalarDist.distance(buf, -1, 0f, 1f, q))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> scalarDist.distance(buf, 1, 0f, 1f, q))
        .isInstanceOf(IndexOutOfBoundsException.class);

    assertThatThrownBy(() -> simdDist.distance(null, 0, 0f, 1f, q))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> simdDist.distance(buf, 0, 0f, 1f, null))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> simdDist.distance(buf, -1, 0f, 1f, q))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> simdDist.distance(buf, 1, 0f, 1f, q))
        .isInstanceOf(IndexOutOfBoundsException.class);
  }
}
