package com.nanovector.core.storage;

import com.nanovector.core.util.VectorUtils;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;

/**
 * Contiguous primitive-based vector storage.
 *
 * <p>Maintains vectors in a single continuous 1D {@code float[]} array:
 *
 * <pre>
 *   offset = internalId * dimension
 * </pre>
 *
 * Manages two-way mapping between external {@code long} IDs and internal {@code int} indices.
 * Rejects duplicate external IDs in Phase 1 with {@link IllegalArgumentException}.
 */
public final class VectorStorage implements VectorDataView {

  private static final int DEFAULT_INITIAL_CAPACITY = 1024;

  private final int dimension;
  private float[] vectors;
  private long[] externalIds;
  private final Map<Long, Integer> externalToInternal;
  private int size;

  public VectorStorage(int dimension) {
    this(dimension, DEFAULT_INITIAL_CAPACITY);
  }

  public VectorStorage(int dimension, int initialCapacity) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (initialCapacity <= 0) {
      throw new IllegalArgumentException("Initial capacity must be positive: " + initialCapacity);
    }
    this.dimension = dimension;
    this.vectors = new float[initialCapacity * dimension];
    this.externalIds = new long[initialCapacity];
    this.externalToInternal = HashMap.newHashMap(initialCapacity);
    this.size = 0;
  }

  /**
   * Constructs a VectorStorage with pre-populated vectors and external ID mappings.
   *
   * @param dimension vector dimensionality
   * @param size number of active vectors
   * @param vectors primitive float buffer containing at least size * dimension elements
   * @param externalIds primitive long buffer containing at least size external IDs
   */
  public VectorStorage(int dimension, int size, float[] vectors, long[] externalIds) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (size < 0) {
      throw new IllegalArgumentException("Size must be non-negative: " + size);
    }
    Objects.requireNonNull(vectors, "Vectors buffer must not be null");
    Objects.requireNonNull(externalIds, "External IDs buffer must not be null");
    if (vectors.length < (long) size * dimension) {
      throw new IllegalArgumentException("Vectors buffer too small for size " + size);
    }
    if (externalIds.length < size) {
      throw new IllegalArgumentException("External IDs buffer too small for size " + size);
    }

    this.dimension = dimension;
    this.size = size;
    this.vectors = vectors;
    this.externalIds = externalIds;
    this.externalToInternal = HashMap.newHashMap(Math.max(16, size));
    for (int i = 0; i < size; i++) {
      long extId = externalIds[i];
      if (this.externalToInternal.putIfAbsent(extId, i) != null) {
        throw new IllegalArgumentException("Duplicate external ID in restored storage: " + extId);
      }
    }
  }

  /**
   * Inserts a vector with an external ID.
   *
   * @param externalId unique external identifier for the vector
   * @param vector the float array vector
   * @return the assigned internal index (0, 1, 2, ...)
   * @throws IllegalArgumentException if vector length mismatch, contains NaN/Inf, or externalId
   *     already exists
   */
  public synchronized int insert(long externalId, float[] vector) {
    VectorUtils.checkDimension(vector, dimension);
    VectorUtils.checkFinite(vector);

    if (externalToInternal.containsKey(externalId)) {
      throw new IllegalArgumentException("Duplicate external ID: " + externalId);
    }

    ensureCapacity(size + 1);

    int internalId = size;
    int offset = internalId * dimension;
    System.arraycopy(vector, 0, vectors, offset, dimension);
    externalIds[internalId] = externalId;
    externalToInternal.put(externalId, internalId);

    size++;
    return internalId;
  }

  /** Returns the external ID corresponding to the given internal index. */
  public long getExternalId(int internalId) {
    checkInternalBounds(internalId);
    return externalIds[internalId];
  }

  /**
   * Finds the internal index for an external ID.
   *
   * @throws NoSuchElementException if externalId does not exist
   */
  public int getInternalId(long externalId) {
    Integer internalId = externalToInternal.get(externalId);
    if (internalId == null) {
      throw new NoSuchElementException("External ID not found: " + externalId);
    }
    return internalId;
  }

  /** Checks if the storage contains the given external ID. */
  public boolean contains(long externalId) {
    return externalToInternal.containsKey(externalId);
  }

  /** Retrieves a copy of the vector at the specified internal index. */
  public float[] getVector(int internalId) {
    checkInternalBounds(internalId);
    float[] copy = new float[dimension];
    System.arraycopy(vectors, internalId * dimension, copy, 0, dimension);
    return copy;
  }

  /**
   * Copies the vector at the specified internal index into a provided destination array, avoiding
   * heap allocations.
   *
   * @param internalId internal index
   * @param dest destination array (must have length == dimension)
   */
  public void copyVector(int internalId, float[] dest) {
    checkInternalBounds(internalId);
    VectorUtils.checkDimension(dest, dimension);
    System.arraycopy(vectors, internalId * dimension, dest, 0, dimension);
  }

  @Override
  public float[] vectorBuffer() {
    return vectors;
  }

  @Override
  public long[] externalIdBuffer() {
    return externalIds;
  }

  /** Computes the element-indexed starting offset for an internal index. */
  public int getOffset(int internalId) {
    checkInternalBounds(internalId);
    return internalId * dimension;
  }

  public int size() {
    return size;
  }

  public int dimension() {
    return dimension;
  }

  public int capacity() {
    return externalIds.length;
  }

  private void ensureCapacity(int minCapacity) {
    int currentCapacity = externalIds.length;
    if (minCapacity > currentCapacity) {
      int newCapacity = Math.max(minCapacity, currentCapacity * 2);
      vectors = Arrays.copyOf(vectors, newCapacity * dimension);
      externalIds = Arrays.copyOf(externalIds, newCapacity);
    }
  }

  private void checkInternalBounds(int internalId) {
    if (internalId < 0 || internalId >= size) {
      throw new IndexOutOfBoundsException(
          "Internal ID out of bounds: " + internalId + ", current size: " + size);
    }
  }
}
