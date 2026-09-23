package com.nanovector.core.distance;

/**
 * Functional interface for vector distance calculations.
 * <p>
 * Implementations must adhere to the <b>Distance Minimization</b> contract:
 * smaller returned values signify higher similarity between vectors.
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
     * Calculates the distance directly from a contiguous vector storage buffer without
     * creating intermediate array copies.
     * <p>
     * <b>Important:</b> {@code offset} is <b>element-indexed</b> (index within the {@code float[]} array),
     * NOT byte-indexed. For example, for dimension = 128:
     * <ul>
     *   <li>Node 0: offset = 0</li>
     *   <li>Node 1: offset = 128</li>
     *   <li>Node 2: offset = 256</li>
     * </ul>
     *
     * @param buffer contiguous primitive array containing stored vectors
     * @param offset starting element index in {@code buffer} where the target vector begins
     * @param query  the query vector
     * @return the calculated distance (smaller is closer/more similar)
     */
    float distance(float[] buffer, int offset, float[] query);

    /**
     * Returns the distance metric type associated with this calculator.
     *
     * @return the {@link DistanceMetric}
     */
    DistanceMetric metric();
}
