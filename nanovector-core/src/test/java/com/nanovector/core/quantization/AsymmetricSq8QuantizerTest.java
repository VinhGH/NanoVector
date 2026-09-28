package com.nanovector.core.quantization;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class AsymmetricSq8QuantizerTest {

  private final AsymmetricSq8Quantizer quantizer = AsymmetricSq8Quantizer.INSTANCE;

  @Test
  @DisplayName(
      "Verify round-trip quantization, theoretical error bound, MSE, and cosine similarity")
  void testQuantizeAndDequantizeRoundTrip() {
    Random rng = new Random(42L);
    int dim = 128;
    int numVectors = 100;

    for (int v = 0; v < numVectors; v++) {
      float[] original = new float[dim];
      for (int i = 0; i < dim; i++) {
        original[i] = rng.nextFloat() * 2.0f - 1.0f; // range [-1, 1]
      }

      QuantizedVector qVec = quantizer.quantize(original);
      assertThat(qVec.dimension()).isEqualTo(dim);
      assertThat(qVec.data()).hasSize(dim);

      float[] reconstructed = quantizer.dequantize(qVec);
      assertThat(reconstructed).hasSize(dim);

      float halfScale = qVec.scale() / 2.0f;
      float sumSquaredError = 0.0f;
      float dotProduct = 0.0f;
      float normOrig = 0.0f;
      float normRecon = 0.0f;

      for (int i = 0; i < dim; i++) {
        float diff = Math.abs(original[i] - reconstructed[i]);
        // Theoretical coordinate error bound: |v_i - \hat{v}_i| <= scale / 2 (+ float precision
        // epsilon)
        assertThat(diff)
            .as("Coordinate %d error must not exceed half the scale step", i)
            .isLessThanOrEqualTo(halfScale + 1e-6f);

        sumSquaredError += diff * diff;
        dotProduct += original[i] * reconstructed[i];
        normOrig += original[i] * original[i];
        normRecon += reconstructed[i] * reconstructed[i];
      }

      // Mean Squared Error (MSE) must be tiny (< 1e-5)
      float mse = sumSquaredError / dim;
      assertThat(mse).isLessThan(1e-5f);

      // Cosine similarity must be exceptionally high (> 0.9999)
      float cosineSimilarity = (float) (dotProduct / (Math.sqrt(normOrig) * Math.sqrt(normRecon)));
      assertThat(cosineSimilarity).isGreaterThan(0.9999f);
    }
  }

  @Test
  @DisplayName("Verify quantization bounds: min maps to 0, max maps to 255")
  void testQuantizationBounds() {
    float[] vector = new float[] {-3.5f, -1.0f, 0.0f, 2.5f, 4.5f};
    QuantizedVector qVec = quantizer.quantize(vector);

    assertThat(qVec.min()).isEqualTo(-3.5f);
    float expectedScale = (4.5f - (-3.5f)) / 255.0f;
    assertThat(qVec.scale()).isEqualTo(expectedScale);

    // Min element (-3.5) must map to 0
    assertThat(qVec.data()[0] & 0xFF).isEqualTo(0);

    // Max element (4.5) must map to 255
    assertThat(qVec.data()[4] & 0xFF).isEqualTo(255);

    // All elements must fall strictly in unsigned range [0, 255]
    for (int i = 0; i < vector.length; i++) {
      int uval = qVec.data()[i] & 0xFF;
      assertThat(uval).isBetween(0, 255);
    }
  }

  @Test
  @DisplayName(
      "Verify constant vector handling: scale=0, all elements map to 0, exact reconstruction")
  void testConstantVector() {
    float[] vector = new float[] {0.75f, 0.75f, 0.75f, 0.75f};
    QuantizedVector qVec = quantizer.quantize(vector);

    assertThat(qVec.min()).isEqualTo(0.75f);
    assertThat(qVec.scale()).isEqualTo(0.0f);

    for (byte b : qVec.data()) {
      assertThat(b).isEqualTo((byte) 0);
    }

    float[] reconstructed = quantizer.dequantize(qVec);
    for (float v : reconstructed) {
      assertThat(v).isEqualTo(0.75f);
    }
  }

  @Test
  @DisplayName("Verify zero vector handling: min=0, scale=0, exact zero reconstruction")
  void testZeroVector() {
    float[] vector = new float[] {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    QuantizedVector qVec = quantizer.quantize(vector);

    assertThat(qVec.min()).isEqualTo(0.0f);
    assertThat(qVec.scale()).isEqualTo(0.0f);

    float[] reconstructed = quantizer.dequantize(qVec);
    for (float v : reconstructed) {
      assertThat(v).isEqualTo(0.0f);
    }
  }

  @Test
  @DisplayName("Verify negative-only and positive-only vectors")
  void testNegativeAndPositiveRanges() {
    // Negative range
    float[] neg = new float[] {-10.0f, -8.0f, -6.0f, -2.0f};
    QuantizedVector qNeg = quantizer.quantize(neg);
    assertThat(qNeg.min()).isEqualTo(-10.0f);
    assertThat(qNeg.data()[0] & 0xFF).isEqualTo(0);
    assertThat(qNeg.data()[3] & 0xFF).isEqualTo(255);

    float[] reconNeg = quantizer.dequantize(qNeg);
    for (int i = 0; i < neg.length; i++) {
      assertThat(Math.abs(neg[i] - reconNeg[i])).isLessThanOrEqualTo(qNeg.scale() / 2.0f + 1e-6f);
    }

    // Positive range
    float[] pos = new float[] {10.0f, 20.0f, 30.0f, 50.0f};
    QuantizedVector qPos = quantizer.quantize(pos);
    assertThat(qPos.min()).isEqualTo(10.0f);
    assertThat(qPos.data()[0] & 0xFF).isEqualTo(0);
    assertThat(qPos.data()[3] & 0xFF).isEqualTo(255);

    float[] reconPos = quantizer.dequantize(qPos);
    for (int i = 0; i < pos.length; i++) {
      assertThat(Math.abs(pos[i] - reconPos[i])).isLessThanOrEqualTo(qPos.scale() / 2.0f + 1e-6f);
    }
  }

  @Test
  @DisplayName("Verify rejection of non-finite values (NaN, Infinity)")
  void testRejectNonFinite() {
    assertThatThrownBy(() -> quantizer.quantize(new float[] {1.0f, Float.NaN, 2.0f}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("non-finite");

    assertThatThrownBy(() -> quantizer.quantize(new float[] {1.0f, Float.POSITIVE_INFINITY, 2.0f}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("non-finite");

    assertThatThrownBy(() -> quantizer.quantize(new float[] {1.0f, Float.NEGATIVE_INFINITY, 2.0f}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("non-finite");
  }

  @Test
  @DisplayName("Verify input validation for null and empty arrays")
  void testInputValidation() {
    assertThatThrownBy(() -> quantizer.quantize(null)).isInstanceOf(NullPointerException.class);

    assertThatThrownBy(() -> quantizer.quantize(new float[0]))
        .isInstanceOf(IllegalArgumentException.class);

    assertThatThrownBy(() -> quantizer.dequantize(null)).isInstanceOf(NullPointerException.class);
  }

  @Test
  @DisplayName("Verify in-place slice quantization matches single-vector quantization exactly")
  void testInPlaceSliceQuantization() {
    int dim = 64;
    int numVecs = 4;
    Random rng = new Random(12345L);

    float[] flatBuffer = new float[numVecs * dim];
    for (int i = 0; i < flatBuffer.length; i++) {
      flatBuffer[i] = rng.nextFloat() * 10.0f - 5.0f;
    }

    byte[] destBuffer = new byte[numVecs * dim];
    QuantizationParams[] params = new QuantizationParams[numVecs];

    // Quantize slices
    for (int v = 0; v < numVecs; v++) {
      params[v] = quantizer.quantize(flatBuffer, v * dim, dim, destBuffer, v * dim);
    }

    // Compare with single-vector quantize
    for (int v = 0; v < numVecs; v++) {
      float[] singleVec = new float[dim];
      System.arraycopy(flatBuffer, v * dim, singleVec, 0, dim);
      QuantizedVector qv = quantizer.quantize(singleVec);

      assertThat(params[v].min()).isEqualTo(qv.min());
      assertThat(params[v].scale()).isEqualTo(qv.scale());

      for (int i = 0; i < dim; i++) {
        assertThat(destBuffer[v * dim + i]).isEqualTo(qv.data()[i]);
      }
    }

    // Dequantize slices in-place
    float[] reconBuffer = new float[numVecs * dim];
    for (int v = 0; v < numVecs; v++) {
      quantizer.dequantize(
          destBuffer, v * dim, dim, params[v].min(), params[v].scale(), reconBuffer, v * dim);
    }

    // Verify reconstruction
    for (int i = 0; i < flatBuffer.length; i++) {
      int vecIdx = i / dim;
      float halfScale = params[vecIdx].scale() / 2.0f;
      assertThat(Math.abs(flatBuffer[i] - reconBuffer[i])).isLessThanOrEqualTo(halfScale + 1e-6f);
    }
  }

  @Test
  @DisplayName("Verify bounds checking on slice methods")
  void testSliceBoundsChecking() {
    float[] src = new float[10];
    byte[] dest = new byte[10];

    assertThatThrownBy(() -> quantizer.quantize(src, 0, 0, dest, 0))
        .isInstanceOf(IllegalArgumentException.class);

    assertThatThrownBy(() -> quantizer.quantize(src, -1, 5, dest, 0))
        .isInstanceOf(IndexOutOfBoundsException.class);

    assertThatThrownBy(() -> quantizer.quantize(src, 0, 5, dest, -1))
        .isInstanceOf(IndexOutOfBoundsException.class);

    assertThatThrownBy(() -> quantizer.quantize(src, 8, 5, dest, 0))
        .isInstanceOf(IndexOutOfBoundsException.class);

    assertThatThrownBy(() -> quantizer.quantize(src, 0, 5, dest, 8))
        .isInstanceOf(IndexOutOfBoundsException.class);

    assertThatThrownBy(() -> quantizer.dequantize(dest, 0, 0, 0f, 1f, src, 0))
        .isInstanceOf(IllegalArgumentException.class);

    assertThatThrownBy(() -> quantizer.dequantize(dest, 8, 5, 0f, 1f, src, 0))
        .isInstanceOf(IndexOutOfBoundsException.class);
  }

  @Test
  @DisplayName("Verify record constructor validation for QuantizationParams and QuantizedVector")
  void testRecordValidation() {
    assertThatThrownBy(() -> new QuantizationParams(Float.NaN, 1.0f))
        .isInstanceOf(IllegalArgumentException.class);

    assertThatThrownBy(() -> new QuantizationParams(0.0f, -0.5f))
        .isInstanceOf(IllegalArgumentException.class);

    assertThatThrownBy(() -> new QuantizedVector(new byte[0], 0.0f, 1.0f))
        .isInstanceOf(IllegalArgumentException.class);

    assertThatThrownBy(() -> new QuantizedVector(new byte[5], 0.0f, -1.0f))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
