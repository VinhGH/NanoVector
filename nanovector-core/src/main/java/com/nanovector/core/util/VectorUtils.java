package com.nanovector.core.util;

import java.util.Objects;

/** Utility functions for vector operations, validations, and normalization. */
public final class VectorUtils {

  /** Epsilon threshold to detect zero or near-zero norms to prevent division by zero. */
  public static final float ZERO_NORM_THRESHOLD = 1e-9f;

  private static final String VECTOR_NOT_NULL_MSG = "Vector must not be null";

  private VectorUtils() {
    // Utility class
  }

  /**
   * Validates that the vector has the expected dimension.
   *
   * @param vector the vector to check
   * @param expectedDimension the expected dimension
   * @throws NullPointerException if vector is null
   * @throws IllegalArgumentException if vector length does not match expectedDimension
   */
  public static void checkDimension(float[] vector, int expectedDimension) {
    Objects.requireNonNull(vector, VECTOR_NOT_NULL_MSG);
    if (vector.length != expectedDimension) {
      throw new IllegalArgumentException(
          "Invalid vector dimension: expected " + expectedDimension + ", but got " + vector.length);
    }
  }

  /**
   * Validates that all elements in the vector are finite (neither NaN nor Infinite).
   *
   * @param vector the vector to check
   * @throws NullPointerException if vector is null
   * @throws IllegalArgumentException if any element is NaN or Infinite
   */
  public static void checkFinite(float[] vector) {
    Objects.requireNonNull(vector, VECTOR_NOT_NULL_MSG);
    for (int i = 0; i < vector.length; i++) {
      if (!Float.isFinite(vector[i])) {
        throw new IllegalArgumentException(
            "Vector contains non-finite value at index " + i + ": " + vector[i]);
      }
    }
  }

  /** Computes the squared Euclidean norm ($L_2^2$) of a vector: $\sum v_i^2$. */
  public static float squaredNorm(float[] vector) {
    Objects.requireNonNull(vector, VECTOR_NOT_NULL_MSG);
    float sum = 0.0f;
    for (float v : vector) {
      sum += v * v;
    }
    return sum;
  }

  /** Computes the Euclidean norm ($L_2$) of a vector: $\sqrt{\sum v_i^2}$. */
  public static float norm(float[] vector) {
    return (float) Math.sqrt(squaredNorm(vector));
  }

  /** Computes the dot product of two vectors: $\sum a_i \cdot b_i$. */
  public static float dotProduct(float[] a, float[] b) {
    checkDimension(b, a.length);
    float dot = 0.0f;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  /**
   * Normalizes a vector to unit length ($L_2$ norm = 1.0) and returns a new array.
   *
   * @param vector the input vector
   * @return a new normalized float array
   * @throws IllegalArgumentException if vector has zero norm (norm <= 1e-9)
   */
  public static float[] normalize(float[] vector) {
    float[] copy = vector.clone();
    normalizeInPlace(copy);
    return copy;
  }

  /**
   * Normalizes a vector in-place to unit length.
   *
   * @param vector the vector to normalize in-place
   * @throws IllegalArgumentException if vector has zero norm (norm <= 1e-9)
   */
  public static void normalizeInPlace(float[] vector) {
    checkFinite(vector);
    float norm = norm(vector);
    if (norm <= ZERO_NORM_THRESHOLD) {
      throw new IllegalArgumentException("Cannot normalize zero-norm vector (norm: " + norm + ")");
    }
    float invNorm = 1.0f / norm;
    for (int i = 0; i < vector.length; i++) {
      vector[i] *= invNorm;
    }
  }
}
