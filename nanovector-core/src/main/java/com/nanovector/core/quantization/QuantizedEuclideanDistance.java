package com.nanovector.core.quantization;

import java.util.Objects;

/**
 * Functional contract for Asymmetric Euclidean Distance (ADC) computation between full-precision
 * FP32 queries and compact 8-bit scalar quantized vectors.
 *
 * <p>Computes squared Euclidean distance ($L_2^2$) directly against contiguous byte buffers without
 * intermediate heap allocations:
 *
 * <pre>
 *   dist(u, q) = \sum_{i=0}^{D-1} (u_i - (min + scale * (q_i & 0xFF)))^2
 * </pre>
 *
 * <p>Distance minimization contract: smaller values represent closer vectors.
 */
public interface QuantizedEuclideanDistance {

  /**
   * Evaluates asymmetric squared Euclidean distance ($L_2^2$) between an FP32 query and a quantized
   * vector slice stored in a byte buffer.
   *
   * @param buffer byte array storing unsigned 8-bit quantized values
   * @param offset starting byte offset where the target quantized vector begins
   * @param min per-vector affine minimum parameter ($v_{\min}$)
   * @param scale per-vector affine scaling step size factor ($\text{scale}$)
   * @param query full-precision FP32 query vector
   * @return squared Euclidean distance ($L_2^2$)
   */
  float distance(byte[] buffer, int offset, float min, float scale, float[] query);

  /**
   * Convenience method to compute asymmetric distance against a standalone {@link QuantizedVector}.
   *
   * @param quantized the quantized vector
   * @param query the full-precision FP32 query vector
   * @return squared Euclidean distance ($L_2^2$)
   */
  default float distance(QuantizedVector quantized, float[] query) {
    Objects.requireNonNull(quantized, "quantized vector must not be null");
    Objects.requireNonNull(query, "query vector must not be null");
    return distance(quantized.data(), 0, quantized.min(), quantized.scale(), query);
  }

  /**
   * Factory method to create an asymmetric Euclidean distance calculator.
   *
   * @param useSimd true for SIMD acceleration via Java Vector API; false for scalar fallback
   * @return distance calculator instance
   */
  static QuantizedEuclideanDistance create(boolean useSimd) {
    return useSimd
        ? new VectorQuantizedEuclideanDistance()
        : new ScalarQuantizedEuclideanDistance();
  }

  /**
   * Factory method to create an asymmetric Euclidean distance calculator with SIMD enabled by
   * default.
   *
   * @return SIMD distance calculator instance
   */
  static QuantizedEuclideanDistance create() {
    return create(true);
  }
}
