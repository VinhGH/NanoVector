package com.nanovector.core.distance;

/**
 * Supported distance metrics for vector similarity search.
 *
 * <p>NanoVector operates strictly as a <b>Distance Minimization Engine</b>: for all metrics, a
 * smaller distance value signifies higher similarity / closer proximity.
 */
public enum DistanceMetric {
  /**
   * Squared Euclidean distance ($L_2^2$): $\sum_{i=0}^{d-1} (a_i - b_i)^2$. The square root is
   * omitted to conserve CPU cycles while preserving relative ranking order.
   */
  EUCLIDEAN,

  /**
   * Cosine distance: $1.0f - (u \cdot v)$ for normalized vectors ($\|u\| = 1, \|v\| = 1$). Vectors
   * are normalized upon insertion and query to eliminate runtime normalization in search loops.
   */
  COSINE,

  /**
   * Inverted Dot Product: $-(u \cdot v)$. Negated so that higher dot product (greater similarity)
   * yields a smaller distance value.
   */
  DOT_PRODUCT
}
