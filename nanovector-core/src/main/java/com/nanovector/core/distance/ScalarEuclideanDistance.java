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
    Objects.requireNonNull(query, "Query vector must not be null");
    return distance(buffer, offset, query, 0, query.length);
  }

  @Override
  public float distance(float[] bufferA, int offsetA, float[] bufferB, int offsetB, int length) {
    Objects.requireNonNull(bufferA, "Buffer A must not be null");
    Objects.requireNonNull(bufferB, "Buffer B must not be null");
    if (offsetA < 0 || offsetA + length > bufferA.length) {
      throw new IndexOutOfBoundsException(
          "OffsetA and length exceed bufferA bounds: offsetA="
              + offsetA
              + ", length="
              + length
              + ", bufferA.length="
              + bufferA.length);
    }
    if (offsetB < 0 || offsetB + length > bufferB.length) {
      throw new IndexOutOfBoundsException(
          "OffsetB and length exceed bufferB bounds: offsetB="
              + offsetB
              + ", length="
              + length
              + ", bufferB.length="
              + bufferB.length);
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
      float diff = bufferA[offsetA + i] - bufferB[offsetB + i];
      sum += diff * diff;
    }
    return sum;
  }

  @Override
  public DistanceMetric metric() {
    return DistanceMetric.EUCLIDEAN;
  }
}
