package com.nanovector.core.distance;

import java.util.Objects;

/**
 * Computes Cosine distance for pre-normalized vectors:
 *
 * <pre>
 *   distance(u, v) = 1.0f - dot(u, v)
 * </pre>
 *
 * Assumes vectors have been normalized to unit length ($\|u\|_2 = 1.0$). Clamps floating-point
 * rounding errors to ensure non-negative distance $\ge 0.0f$.
 */
public final class ScalarCosineDistance implements DistanceCalculator {

  @Override
  public float distance(float[] a, float[] b) {
    Objects.requireNonNull(a, "Vector 'a' must not be null");
    Objects.requireNonNull(b, "Vector 'b' must not be null");
    if (a.length != b.length) {
      throw new IllegalArgumentException(
          "Vector dimension mismatch: a.length=" + a.length + ", b.length=" + b.length);
    }

    float dot = 0.0f;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return Math.max(0.0f, 1.0f - dot);
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

    float dot = 0.0f;
    for (int i = 0; i < length; i++) {
      dot += bufferA[offsetA + i] * bufferB[offsetB + i];
    }
    return Math.max(0.0f, 1.0f - dot);
  }

  @Override
  public DistanceMetric metric() {
    return DistanceMetric.COSINE;
  }
}
