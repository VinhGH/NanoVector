package com.nanovector.core.quantization;

import java.util.Objects;

/**
 * Loop-fused single-pass scalar calculation for Asymmetric Euclidean Distance (ADC).
 *
 * <p>Evaluates squared Euclidean distance directly from contiguous byte buffer storage on the fly
 * without dequantizing into intermediate arrays, producing zero heap allocations in the inner loop:
 *
 * <pre>
 *   dist(u, q) = \sum_{i=0}^{D-1} (u_i - (min + scale * (q_i & 0xFF)))^2
 * </pre>
 */
public final class ScalarQuantizedEuclideanDistance implements QuantizedEuclideanDistance {

  @Override
  public float distance(byte[] buffer, int offset, float min, float scale, float[] query) {
    Objects.requireNonNull(buffer, "buffer must not be null");
    Objects.requireNonNull(query, "query must not be null");
    int length = query.length;
    if (offset < 0 || offset + length > buffer.length) {
      throw new IndexOutOfBoundsException(
          String.format(
              "Buffer slice out of bounds: offset=%d, length=%d, buffer.length=%d",
              offset, length, buffer.length));
    }

    if (scale == 0.0f) {
      float sum = 0.0f;
      for (int i = 0; i < length; i++) {
        float diff = query[i] - min;
        sum += diff * diff;
      }
      return sum;
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
      int q = buffer[offset + i] & 0xFF;
      float recon = min + scale * q;
      float diff = query[i] - recon;
      sum += diff * diff;
    }
    return sum;
  }
}
