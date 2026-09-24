package com.nanovector.core.distance;

/**
 * Functional interface for vector distance calculations.
 *
 * <p>Implementations must adhere to the <b>Distance Minimization</b> contract: smaller returned
 * values signify higher similarity between vectors.
 */
public interface DistanceCalculator {

  /**
   * Calculates the distance between two standalone vectors.
   *
   * @param a the first vector
   * @param b the second vector
   * @return the calculated distance (smaller is closer/more similar)
   */
  float distance(float[] a, float[] b);

  /**
   * Calculates the distance directly from a contiguous vector storage buffer without creating
   * intermediate array copies.
   *
   * <p><b>Important:</b> {@code offset} is <b>element-indexed</b> (index within the {@code float[]}
   * array), NOT byte-indexed. For example, for dimension = 128:
   *
   * <ul>
   *   <li>Node 0: offset = 0
   *   <li>Node 1: offset = 128
   *   <li>Node 2: offset = 256
   * </ul>
   *
   * @param buffer contiguous primitive array containing stored vectors
   * @param offset starting element index in {@code buffer} where the target vector begins
   * @param query the query vector
   * @return the calculated distance (smaller is closer/more similar)
   */
  float distance(float[] buffer, int offset, float[] query);

  /**
   * Calculates the distance directly between two vectors located within primitive storage buffers
   * without allocating intermediate arrays.
   *
   * @param bufferA primitive float array containing vector A
   * @param offsetA element-indexed starting position of vector A
   * @param bufferB primitive float array containing vector B
   * @param offsetB element-indexed starting position of vector B
   * @param length number of vector dimensions to evaluate
   * @return the calculated distance (smaller is closer/more similar)
   */
  float distance(float[] bufferA, int offsetA, float[] bufferB, int offsetB, int length);

  /**
   * Calculates the distance directly between two vectors located within the same primitive storage
   * buffer without allocating intermediate arrays.
   *
   * @param buffer contiguous primitive array containing stored vectors
   * @param offsetA element-indexed starting position of vector A
   * @param offsetB element-indexed starting position of vector B
   * @param length number of vector dimensions to evaluate
   * @return the calculated distance (smaller is closer/more similar)
   */
  default float distance(float[] buffer, int offsetA, int offsetB, int length) {
    return distance(buffer, offsetA, buffer, offsetB, length);
  }

  /**
   * Returns the distance metric type associated with this calculator.
   *
   * @return the {@link DistanceMetric}
   */
  DistanceMetric metric();
}
