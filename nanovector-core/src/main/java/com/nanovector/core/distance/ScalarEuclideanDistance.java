package com.nanovector.core.distance;

import java.util.Objects;

/**
 * Computes squared Euclidean distance ($L_2^2$):
 *
 * <pre>
 *   distance(a, b) = sum((a[i] - b[i])^2)
 * </pre>
 *
 * Omits the square root operation to preserve CPU cycles while maintaining relative order.
 */
public final class ScalarEuclideanDistance implements DistanceCalculator {

  @Override
  public float distance(float[] a, float[] b) {
    Objects.requireNonNull(a, "Vector 'a' must not be null");
    Objects.requireNonNull(b, "Vector 'b' must not be null");
    if (a.length != b.length) {
      throw new IllegalArgumentException(
          "Vector dimension mismatch: a.length=" + a.length + ", b.length=" + b.length);
    }

    float sum = 0.0f;
    for (int i = 0; i < a.length; i++) {
      float diff = a[i] - b[i];
      sum += diff * diff;
    }
    return sum;
  }

  @Override
  public float distance(float[] buffer, int offset, float[] query) {
    Objects.requireNonNull(buffer, "Buffer must not be null");
    Objects.requireNonNull(query, "Query vector must not be null");
    if (offset < 0 || offset + query.length > buffer.length) {
      throw new IndexOutOfBoundsException(
          "Offset and query length exceed buffer bounds: offset="
              + offset
              + ", query.length="
              + query.length
              + ", buffer.length="
              + buffer.length);
    }

    float sum = 0.0f;
    for (int i = 0; i < query.length; i++) {
      float diff = buffer[offset + i] - query[i];
      sum += diff * diff;
    }
    return sum;
  }

  @Override
  public DistanceMetric metric() {
    return DistanceMetric.EUCLIDEAN;
  }
}
