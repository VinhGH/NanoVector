package com.nanovector.core.index;

import com.nanovector.core.distance.DistanceCalculator;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.heap.BoundedMaxHeap;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.storage.VectorStorage;
import com.nanovector.core.util.VectorUtils;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Exact brute-force vector index ($O(N)$ sequential scan).
 *
 * <p>Serves as the <b>Ground Truth Oracle</b> for measuring recall and verifying approximate
 * nearest neighbor (ANN) indexes.
 *
 * <p>Key performance characteristic: Evaluates distances directly against the contiguous primitive
 * buffer of {@link VectorStorage}, producing zero allocations inside the $N$-element distance scan
 * loop.
 */
public final class FlatIndex implements VectorIndex {

  private final int dimension;
  private final DistanceMetric metric;
  private final DistanceCalculator calculator;
  private final VectorStorage storage;

  public FlatIndex(int dimension, DistanceMetric metric) {
    this(dimension, metric, 1024, false);
  }

  public FlatIndex(int dimension, DistanceMetric metric, boolean useSimd) {
    this(dimension, metric, 1024, useSimd);
  }

  public FlatIndex(int dimension, DistanceMetric metric, int initialCapacity) {
    this(dimension, metric, initialCapacity, false);
  }

  public FlatIndex(int dimension, DistanceMetric metric, int initialCapacity, boolean useSimd) {
    this(dimension, metric, initialCapacity, DistanceCalculator.create(metric, useSimd));
  }

  public FlatIndex(
      int dimension, DistanceMetric metric, int initialCapacity, DistanceCalculator calculator) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    this.dimension = dimension;
    this.metric = Objects.requireNonNull(metric, "Metric must not be null");
    this.calculator = Objects.requireNonNull(calculator, "Calculator must not be null");
    this.storage = new VectorStorage(dimension, initialCapacity);
  }

  /** Returns the distance calculator used by this index. */
  public DistanceCalculator calculator() {
    return calculator;
  }

  @Override
  public synchronized void insert(long id, float[] vector) {
    if (metric == DistanceMetric.COSINE) {
      float[] normalized = VectorUtils.normalize(vector);
      storage.insert(id, normalized);
    } else {
      storage.insert(id, vector);
    }
  }

  @Override
  public List<SearchResult> searchKnn(float[] query, int k) {
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    VectorUtils.checkDimension(query, dimension);
    VectorUtils.checkFinite(query);

    int total = storage.size();
    if (total == 0) {
      return Collections.emptyList();
    }

    float[] effectiveQuery;
    if (metric == DistanceMetric.COSINE) {
      effectiveQuery = VectorUtils.normalize(query);
    } else {
      effectiveQuery = query;
    }

    int actualK = Math.min(k, total);
    BoundedMaxHeap heap = new BoundedMaxHeap(actualK);

    float[] buffer = storage.getVectorBuffer();
    for (int i = 0; i < total; i++) {
      int offset = i * dimension;
      float dist = calculator.distance(buffer, offset, effectiveQuery);
      long externalId = storage.getExternalId(i);
      heap.offer(externalId, dist);
    }

    return heap.toSortedList();
  }

  @Override
  public int size() {
    return storage.size();
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public DistanceMetric metric() {
    return metric;
  }

  /** Retrieves stored vector copy for inspection/testing. */
  public float[] getVector(int internalId) {
    return storage.getVector(internalId);
  }

  /** Direct reference to storage (internal access). */
  VectorStorage getStorage() {
    return storage;
  }
}
