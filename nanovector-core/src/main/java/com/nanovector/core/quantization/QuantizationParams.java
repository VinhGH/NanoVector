package com.nanovector.core.quantization;

/**
 * Per-vector affine quantization parameters containing the minimum value and step size scale.
 *
 * <p>Quantization granularity: <b>per-vector</b>.
 *
 * @param min the minimum coordinate value of the original vector
 * @param scale the quantization step size factor ({@code (max - min) / 255.0f})
 */
public record QuantizationParams(float min, float scale) {

  public QuantizationParams {
    if (!Float.isFinite(min)) {
      throw new IllegalArgumentException("min must be a finite float, got: " + min);
    }
    if (!Float.isFinite(scale)) {
      throw new IllegalArgumentException("scale must be a finite float, got: " + scale);
    }
    if (scale < 0.0f) {
      throw new IllegalArgumentException("scale must be non-negative, got: " + scale);
    }
  }
}
