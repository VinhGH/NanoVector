package com.nanovector.core.quantization;

/**
 * Contract for vector scalar quantization, converting full-precision FP32 vectors into compact
 * 8-bit representations and reconstructing them back.
 *
 * <p>Quantization granularity: <b>per-vector</b>. Stored values are unsigned 8-bit integers mapped
 * to Java {@code byte[]} arrays.
 */
public interface ScalarQuantizer {

  /**
   * Quantizes a single full-precision FP32 vector into a {@link QuantizedVector}.
   *
   * @param vector the source float vector
   * @return the quantized vector containing byte array and affine parameters
   * @throws NullPointerException if vector is null
   * @throws IllegalArgumentException if vector contains non-finite values (NaN, Infinity) or is
   *     empty
   */
  QuantizedVector quantize(float[] vector);

  /**
   * Reconstructs a full-precision float array from a {@link QuantizedVector}.
   *
   * @param quantized the quantized vector
   * @return reconstructed float array
   * @throws NullPointerException if quantized is null
   */
  float[] dequantize(QuantizedVector quantized);

  /**
   * Quantizes an input float slice into a destination byte array slice in-place, returning the
   * affine parameters ({@code min}, {@code scale}) to avoid intermediate array allocations.
   *
   * @param src source float array
   * @param srcOffset starting offset in src
   * @param length vector dimension
   * @param dest destination byte array for quantized unsigned 8-bit values
   * @param destOffset starting offset in dest
   * @return per-vector affine quantization parameters (min, scale)
   * @throws NullPointerException if src or dest is null
   * @throws IllegalArgumentException if length <= 0 or non-finite values encountered
   * @throws IndexOutOfBoundsException if offsets/length exceed array bounds
   */
  QuantizationParams quantize(float[] src, int srcOffset, int length, byte[] dest, int destOffset);

  /**
   * Dequantizes unsigned 8-bit values from a byte buffer slice directly into a destination float
   * array slice in-place.
   *
   * @param src source byte array storing unsigned 8-bit values
   * @param srcOffset starting offset in src
   * @param length vector dimension
   * @param min minimum value used during affine quantization
   * @param scale scaling factor used during affine quantization
   * @param dest destination float array
   * @param destOffset starting offset in dest
   * @throws NullPointerException if src or dest is null
   * @throws IllegalArgumentException if length <= 0
   * @throws IndexOutOfBoundsException if offsets/length exceed array bounds
   */
  void dequantize(
      byte[] src, int srcOffset, int length, float min, float scale, float[] dest, int destOffset);
}
