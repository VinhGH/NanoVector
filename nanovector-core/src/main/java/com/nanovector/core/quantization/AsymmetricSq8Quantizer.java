package com.nanovector.core.quantization;

import java.util.Objects;

/**
 * Asymmetric 8-bit Scalar Quantizer (SQ8) with <b>per-vector</b> granularity.
 *
 * <p>Mathematical Specification:
 *
 * <ul>
 *   <li><b>Quantization Granularity</b>: Per-vector. Each vector $\mathbf{v} \in \mathbb{R}^D$ is
 *       quantized independently using its own coordinate bounds: $$v_{\min} = \min_{0 \le i < D}
 *       v_i, \quad v_{\max} = \max_{0 \le i < D} v_i$$
 *   <li><b>Step Size Factor</b>: $$\text{scale} = \frac{v_{\max} - v_{\min}}{255.0f}$$ If $v_{\max}
 *       == v_{\min}$ (constant or zero-range vector), $\text{scale} = 0.0f$ and all quantized
 *       coordinates are assigned $0$.
 *   <li><b>Quantization Mapping (Affine)</b>: $$q_i = \text{clamp}\left(\text{round}\left(\frac{v_i
 *       - v_{\min}}{\text{scale}}\right), 0, 255\right)$$
 *   <li><b>Storage Representation</b>: Unsigned 8-bit values mapped to Java {@code byte} where
 *       decoded unsigned integer $q_i \in [0, 255]$ is retrieved via {@code (byteValue & 0xFF)}.
 *   <li><b>Dequantization (Reconstruction)</b>: $$\hat{v}_i = v_{\min} + \text{scale} \cdot (q_i \
 *       \& \ 0\text{xFF})$$
 *   <li><b>Theoretical Error Bound</b>: Rounding to the nearest integer guarantees coordinate-wise
 *       error $|v_i - \hat{v}_i| \le \frac{\text{scale}}{2} = \frac{v_{\max} - v_{\min}}{510}$.
 * </ul>
 */
public final class AsymmetricSq8Quantizer implements ScalarQuantizer {

  /** Singleton instance for general use. */
  public static final AsymmetricSq8Quantizer INSTANCE = new AsymmetricSq8Quantizer();

  private static final float NUM_LEVELS = 255.0f;

  public AsymmetricSq8Quantizer() {
    // Public default constructor
  }

  @Override
  public QuantizedVector quantize(float[] vector) {
    Objects.requireNonNull(vector, "vector must not be null");
    if (vector.length == 0) {
      throw new IllegalArgumentException("Vector must not be empty");
    }
    byte[] data = new byte[vector.length];
    QuantizationParams params = quantize(vector, 0, vector.length, data, 0);
    return new QuantizedVector(data, params.min(), params.scale());
  }

  @Override
  public float[] dequantize(QuantizedVector quantized) {
    Objects.requireNonNull(quantized, "quantized vector must not be null");
    float[] reconstructed = new float[quantized.dimension()];
    dequantize(
        quantized.data(),
        0,
        quantized.dimension(),
        quantized.min(),
        quantized.scale(),
        reconstructed,
        0);
    return reconstructed;
  }

  @Override
  public QuantizationParams quantize(
      float[] src, int srcOffset, int length, byte[] dest, int destOffset) {
    Objects.requireNonNull(src, "src array must not be null");
    Objects.requireNonNull(dest, "dest array must not be null");
    if (length <= 0) {
      throw new IllegalArgumentException("Vector dimension length must be > 0, got: " + length);
    }
    if (srcOffset < 0 || srcOffset + length > src.length) {
      throw new IndexOutOfBoundsException(
          String.format(
              "Source slice out of bounds: srcOffset=%d, length=%d, src.length=%d",
              srcOffset, length, src.length));
    }
    if (destOffset < 0 || destOffset + length > dest.length) {
      throw new IndexOutOfBoundsException(
          String.format(
              "Destination slice out of bounds: destOffset=%d, length=%d, dest.length=%d",
              destOffset, length, dest.length));
    }

    float min = Float.POSITIVE_INFINITY;
    float max = Float.NEGATIVE_INFINITY;

    for (int i = 0; i < length; i++) {
      float v = src[srcOffset + i];
      if (!Float.isFinite(v)) {
        throw new IllegalArgumentException(
            "Vector contains non-finite value at index " + (srcOffset + i) + ": " + v);
      }
      if (v < min) {
        min = v;
      }
      if (v > max) {
        max = v;
      }
    }

    if (max == min) {
      for (int i = 0; i < length; i++) {
        dest[destOffset + i] = 0;
      }
      return new QuantizationParams(min, 0.0f);
    }

    float scale = (max - min) / NUM_LEVELS;
    float invScale = 1.0f / scale;

    for (int i = 0; i < length; i++) {
      float v = src[srcOffset + i];
      int q = Math.round((v - min) * invScale);
      if (q < 0) {
        q = 0;
      } else if (q > 255) {
        q = 255;
      }
      dest[destOffset + i] = (byte) q;
    }

    return new QuantizationParams(min, scale);
  }

  @Override
  public void dequantize(
      byte[] src, int srcOffset, int length, float min, float scale, float[] dest, int destOffset) {
    Objects.requireNonNull(src, "src array must not be null");
    Objects.requireNonNull(dest, "dest array must not be null");
    if (length <= 0) {
      throw new IllegalArgumentException("Vector dimension length must be > 0, got: " + length);
    }
    if (srcOffset < 0 || srcOffset + length > src.length) {
      throw new IndexOutOfBoundsException(
          String.format(
              "Source slice out of bounds: srcOffset=%d, length=%d, src.length=%d",
              srcOffset, length, src.length));
    }
    if (destOffset < 0 || destOffset + length > dest.length) {
      throw new IndexOutOfBoundsException(
          String.format(
              "Destination slice out of bounds: destOffset=%d, length=%d, dest.length=%d",
              destOffset, length, dest.length));
    }

    if (scale == 0.0f) {
      for (int i = 0; i < length; i++) {
        dest[destOffset + i] = min;
      }
      return;
    }

    for (int i = 0; i < length; i++) {
      int q = src[srcOffset + i] & 0xFF;
      dest[destOffset + i] = min + scale * q;
    }
  }
}
