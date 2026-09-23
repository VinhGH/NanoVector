package com.nanovector.core.index;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.model.SearchResult;

import java.util.List;

/**
 * Common contract for vector similarity search indexes.
 */
public interface VectorIndex {

    /**
     * Inserts a vector with a unique external identifier into the index.
     *
     * @param id     unique external ID
     * @param vector float array vector matching the index dimension
     * @throws IllegalArgumentException if vector length mismatch, contains NaN/Inf, or ID exists
     */
    void insert(long id, float[] vector);

    /**
     * Finds the Top-K nearest neighbors to the query vector.
     *
     * @param query query vector matching the index dimension
     * @param k     number of nearest neighbors to return (must be > 0)
     * @return sorted list of {@link SearchResult} ordered by distance ascending
     */
    List<SearchResult> searchKnn(float[] query, int k);

    /**
     * Returns the total number of indexed vectors.
     */
    int size();

    /**
     * Returns the dimensionality of vectors in this index.
     */
    int dimension();

    /**
     * Returns the distance metric used by this index.
     */
    DistanceMetric metric();
}
