package com.nanovector.core.distance;

import java.util.Objects;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-accelerated Cosine distance for pre-normalized unit vectors using the Java Vector API
 * ({@code jdk.incubator.vector}).
 *
 * <pre>
 *   distance(u, v) = 1.0f - dot(u, v)
 * </pre>
 *
 * <p>Assumes vectors are already normalized upon insertion and query ingestion. Computes inner
 * products across SIMD lanes via {@link FloatVector#SPECIES_PREFERRED} and clamps floating-point
 * rounding artifacts to ensure non-negative distance $\ge 0.0f$.
 */
public final class VectorCosineDistance implements DistanceCalculator {

  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  @Override
  public float distance(float[] a, float[] b) {
    Objects.requireNonNull(a, "Vector 'a' must not be null");
    Objects.requireNonNull(b, "Vector 'b' must not be null");
    if (a.length != b.length) {
      throw new IllegalArgumentException(
          "Vector dimension mismatch: a.length=" + a.length + ", b.length=" + b.length);
    }
    return distance(a, 0, b, 0, a.length);
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
    if (length < 0) {
      throw new IllegalArgumentException("Length must be non-negative: " + length);
    }
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

    int upperBound = SPECIES.loopBound(length);
    FloatVector sumVec = FloatVector.zero(SPECIES);

    int i = 0;
    for (; i < upperBound; i += SPECIES.length()) {
      FloatVector va = FloatVector.fromArray(SPECIES, bufferA, offsetA + i);
      FloatVector vb = FloatVector.fromArray(SPECIES, bufferB, offsetB + i);
      sumVec = va.fma(vb, sumVec);
    }

    float dot = sumVec.reduceLanes(VectorOperators.ADD);

    // Scalar tail loop for remaining dimensions
    for (; i < length; i++) {
      dot += bufferA[offsetA + i] * bufferB[offsetB + i];
    }

    return Math.max(0.0f, 1.0f - dot);
  }

  @Override
  public DistanceMetric metric() {
    return DistanceMetric.COSINE;
  }
}
