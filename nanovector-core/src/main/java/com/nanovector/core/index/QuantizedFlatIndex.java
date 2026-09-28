package com.nanovector.core.index;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.heap.BoundedMaxHeap;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.quantization.QuantizedEuclideanDistance;
import com.nanovector.core.quantization.QuantizedVectorStorage;
import com.nanovector.core.storage.VectorDataView;
import com.nanovector.core.util.VectorUtils;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Approximate flat vector index using 8-bit scalar quantization (SQ8) with Asymmetric Distance
 * Computation (ADC).
 *
 * <p>Serves as the <b>SQ8 approximate flat baseline</b> for Phase 6B: isolates quantization error
 * from graph routing by performing an exhaustive $O(N)$ sequential scan directly against the
 * contiguous byte buffer of {@link QuantizedVectorStorage}.
 *
 * <p>Distances are evaluated on the fly without dequantizing into intermediate arrays, producing
 * zero allocations inside the $N$-element distance scan loop.
 */
public final class QuantizedFlatIndex implements VectorIndex {

  private final int dimension;
  private final DistanceMetric metric;
  private final QuantizedEuclideanDistance calculator;
  private final QuantizedVectorStorage storage;

  public QuantizedFlatIndex(int dimension) {
    this(dimension, 1024, true);
  }

  public QuantizedFlatIndex(int dimension, boolean useSimd) {
    this(dimension, 1024, useSimd);
  }

  public QuantizedFlatIndex(int dimension, int initialCapacity) {
    this(dimension, initialCapacity, true);
  }

  public QuantizedFlatIndex(int dimension, int initialCapacity, boolean useSimd) {
    this(dimension, initialCapacity, QuantizedEuclideanDistance.create(useSimd));
  }

  public QuantizedFlatIndex(
      int dimension, int initialCapacity, QuantizedEuclideanDistance calculator) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    this.dimension = dimension;
    this.metric = DistanceMetric.EUCLIDEAN;
    this.calculator = Objects.requireNonNull(calculator, "Calculator must not be null");
    this.storage = new QuantizedVectorStorage(dimension, initialCapacity);
  }

  public QuantizedFlatIndex(
      int dimension, DistanceMetric metric, int initialCapacity, boolean useSimd) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (metric != DistanceMetric.EUCLIDEAN) {
      throw new UnsupportedOperationException(
          "QuantizedFlatIndex currently supports EUCLIDEAN metric only, got: " + metric);
    }
    this.dimension = dimension;
    this.metric = metric;
    this.calculator = QuantizedEuclideanDistance.create(useSimd);
    this.storage = new QuantizedVectorStorage(dimension, initialCapacity);
  }

  public QuantizedFlatIndex(
      int dimension, QuantizedVectorStorage storage, QuantizedEuclideanDistance calculator) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    this.dimension = dimension;
    this.metric = DistanceMetric.EUCLIDEAN;
    this.storage = Objects.requireNonNull(storage, "Storage must not be null");
    this.calculator = Objects.requireNonNull(calculator, "Calculator must not be null");
  }

  /** Returns the asymmetric distance calculator used by this index. */
  public QuantizedEuclideanDistance calculator() {
    return calculator;
  }

  /** Returns the underlying quantized vector storage. */
  public QuantizedVectorStorage storage() {
    return storage;
  }

  @Override
  public synchronized void insert(long id, float[] vector) {
    storage.insert(id, vector);
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

    int actualK = Math.min(k, total);
    BoundedMaxHeap heap = new BoundedMaxHeap(actualK);

    byte[] buffer = storage.vectorBuffer();
    float[] mins = storage.minBuffer();
    float[] scales = storage.scaleBuffer();

    for (int i = 0; i < total; i++) {
      int offset = i * dimension;
      float dist = calculator.distance(buffer, offset, mins[i], scales[i], query);
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

  @Override
  public VectorDataView vectorData() {
    throw new UnsupportedOperationException(
        "QuantizedFlatIndex maintains contiguous quantized byte storage. Use storage() to access QuantizedVectorStorage.");
  }

  /** Retrieves a dequantized float vector copy for testing and inspection. */
  public float[] getVector(int internalId) {
    return storage.getVector(internalId);
  }
}
