package com.nanovector.core.offheap;

import java.lang.foreign.MemorySegment;

/**
 * Functional contract for Asymmetric Euclidean Distance (ADC) computation between full-precision
 * FP32 queries and 8-bit scalar quantized vectors stored directly in native {@link MemorySegment}s.
 *
 * <p>Computes squared Euclidean distance ($L_2^2$) without copying bytes into Java heap arrays:
 *
 * <pre>
 *   dist(u, q) = \sum_{i=0}^{D-1} (u_i - (min + scale * (q_i & 0xFF)))^2
 * </pre>
 */
public interface OffHeapQuantizedEuclideanDistance {

  /**
   * Evaluates asymmetric squared Euclidean distance ($L_2^2$) between an FP32 query and a quantized
   * vector stored in a native memory segment.
   *
   * @param segment native memory segment containing quantized coordinates
   * @param offset byte offset where the target vector's quantized coordinates begin
   * @param min per-vector affine minimum parameter ($v_{\min}$)
   * @param scale per-vector affine scaling step size factor ($\text{scale}$)
   * @param query full-precision FP32 query vector
   * @return squared Euclidean distance ($L_2^2$)
   */
  float distance(MemorySegment segment, long offset, float min, float scale, float[] query);

  /**
   * Factory method to create an off-heap asymmetric Euclidean distance calculator.
   *
   * @param useSimd true for SIMD Vector API acceleration; false for scalar fallback
   * @return distance calculator instance
   */
  static OffHeapQuantizedEuclideanDistance create(boolean useSimd) {
    return useSimd
        ? new OffHeapVectorQuantizedEuclideanDistance()
        : new OffHeapScalarQuantizedEuclideanDistance();
  }

  /**
   * Factory method to create an off-heap asymmetric Euclidean distance calculator with SIMD enabled
   * by default.
   */
  static OffHeapQuantizedEuclideanDistance create() {
    return create(true);
  }
}
