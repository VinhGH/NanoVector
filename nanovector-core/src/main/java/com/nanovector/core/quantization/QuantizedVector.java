package com.nanovector.core.quantization;

import java.util.Objects;

/**
 * An immutable representation of a single vector quantized to unsigned 8-bit values stored in a
 * Java {@code byte[]} array, accompanied by per-vector affine quantization parameters ({@code min}
 * and {@code scale}).
 *
 * <p>Quantization granularity: <b>per-vector</b>. Stored values are unsigned 8-bit integers in the
 * range {@code [0, 255]}, decoded via {@code (byteValue & 0xFF)}.
 *
 * @param data byte array storing unsigned 8-bit values
 * @param min the minimum coordinate value of the original vector before quantization
 * @param scale the quantization step size factor ({@code (max - min) / 255.0f})
 */
public record QuantizedVector(byte[] data, float min, float scale) {

  public QuantizedVector {
    Objects.requireNonNull(data, "data array must not be null");
    if (data.length == 0) {
      throw new IllegalArgumentException("data array must not be empty");
    }
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

  /**
   * Returns the vector dimension.
   *
   * @return vector length
   */
  public int dimension() {
    return data.length;
  }
}
