package com.nanovector.core.quantization;

import com.nanovector.core.util.VectorUtils;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;

/**
 * Contiguous primitive storage with allocation-free vector access for 8-bit scalar quantized
 * vectors.
 *
 * <p>Maintains quantized unsigned 8-bit values in a single continuous 1D {@code byte[]} array,
 * alongside parallel primitive float arrays for per-vector affine parameters ($v_{\min}$ and
 * $\text{scale}$):
 *
 * <pre>
 *   offset = internalId * dimension
 * </pre>
 *
 * <p>Vector memory footprint: $D$ bytes (quantized coordinates) + 4 bytes ($v_{\min}$) + 4 bytes
 * ($\text{scale}$) = $D + 8$ bytes per vector. For $D=128$, this consumes 136 bytes per vector
 * compared to 512 bytes for full-precision FP32, achieving a $\approx 3.76\times$ reduction in
 * vector data memory.
 */
public final class QuantizedVectorStorage {

  private static final int DEFAULT_INITIAL_CAPACITY = 1024;

  private final int dimension;
  private final ScalarQuantizer quantizer;
  private byte[] vectors;
  private float[] mins;
  private float[] scales;
  private long[] externalIds;
  private final Map<Long, Integer> externalToInternal;
  private int size;

  public QuantizedVectorStorage(int dimension) {
    this(dimension, DEFAULT_INITIAL_CAPACITY, AsymmetricSq8Quantizer.INSTANCE);
  }

  public QuantizedVectorStorage(int dimension, int initialCapacity) {
    this(dimension, initialCapacity, AsymmetricSq8Quantizer.INSTANCE);
  }

  public QuantizedVectorStorage(int dimension, int initialCapacity, ScalarQuantizer quantizer) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (initialCapacity <= 0) {
      throw new IllegalArgumentException("Initial capacity must be positive: " + initialCapacity);
    }
    this.dimension = dimension;
    this.quantizer = Objects.requireNonNull(quantizer, "quantizer must not be null");
    this.vectors = new byte[initialCapacity * dimension];
    this.mins = new float[initialCapacity];
    this.scales = new float[initialCapacity];
    this.externalIds = new long[initialCapacity];
    this.externalToInternal = HashMap.newHashMap(initialCapacity);
    this.size = 0;
  }

  /**
   * Constructs a QuantizedVectorStorage with pre-populated vectors, parameters, and external ID
   * mappings.
   *
   * @param dimension vector dimensionality
   * @param size number of active vectors
   * @param vectors contiguous byte buffer containing at least size * dimension elements
   * @param mins primitive float array containing at least size min parameters
   * @param scales primitive float array containing at least size scale parameters
   * @param externalIds primitive long array containing at least size external IDs
   */
  public QuantizedVectorStorage(
      int dimension, int size, byte[] vectors, float[] mins, float[] scales, long[] externalIds) {
    this(dimension, size, vectors, mins, scales, externalIds, AsymmetricSq8Quantizer.INSTANCE);
  }

  /** Constructs a QuantizedVectorStorage with pre-populated buffers and custom quantizer. */
  public QuantizedVectorStorage(
      int dimension,
      int size,
      byte[] vectors,
      float[] mins,
      float[] scales,
      long[] externalIds,
      ScalarQuantizer quantizer) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (size < 0) {
      throw new IllegalArgumentException("Size must be non-negative: " + size);
    }
    Objects.requireNonNull(vectors, "Vectors buffer must not be null");
    Objects.requireNonNull(mins, "Mins buffer must not be null");
    Objects.requireNonNull(scales, "Scales buffer must not be null");
    Objects.requireNonNull(externalIds, "External IDs buffer must not be null");
    this.quantizer = Objects.requireNonNull(quantizer, "quantizer must not be null");

    if (vectors.length < (long) size * dimension) {
      throw new IllegalArgumentException("Vectors buffer too small for size " + size);
    }
    if (mins.length < size) {
      throw new IllegalArgumentException("Mins buffer too small for size " + size);
    }
    if (scales.length < size) {
      throw new IllegalArgumentException("Scales buffer too small for size " + size);
    }
    if (externalIds.length < size) {
      throw new IllegalArgumentException("External IDs buffer too small for size " + size);
    }

    this.dimension = dimension;
    this.size = size;
    this.vectors = vectors;
    this.mins = mins;
    this.scales = scales;
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
   * Inserts an FP32 vector with a unique external ID, quantizing it in-place into the contiguous
   * byte buffer.
   *
   * @param externalId unique external identifier for the vector
   * @param vector the full-precision FP32 vector
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
    QuantizationParams params = quantizer.quantize(vector, 0, dimension, vectors, offset);

    mins[internalId] = params.min();
    scales[internalId] = params.scale();
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

  /** Computes the element-indexed starting byte offset for an internal index. */
  public int getOffset(int internalId) {
    checkInternalBounds(internalId);
    return internalId * dimension;
  }

  /** Retrieves the affine minimum parameter ($v_{\min}$) for an internal index. */
  public float getMin(int internalId) {
    checkInternalBounds(internalId);
    return mins[internalId];
  }

  /** Retrieves the affine step size scaling factor ($\text{scale}$) for an internal index. */
  public float getScale(int internalId) {
    checkInternalBounds(internalId);
    return scales[internalId];
  }

  /** Reconstructs and returns an FP32 copy of the vector at the specified internal index. */
  public float[] getVector(int internalId) {
    checkInternalBounds(internalId);
    float[] reconstructed = new float[dimension];
    quantizer.dequantize(
        vectors,
        internalId * dimension,
        dimension,
        mins[internalId],
        scales[internalId],
        reconstructed,
        0);
    return reconstructed;
  }

  /**
   * Reconstructs the vector at the specified internal index into the provided destination array,
   * avoiding heap allocations.
   *
   * @param internalId internal index
   * @param dest destination array (must have length == dimension)
   */
  public void copyVector(int internalId, float[] dest) {
    checkInternalBounds(internalId);
    VectorUtils.checkDimension(dest, dimension);
    quantizer.dequantize(
        vectors, internalId * dimension, dimension, mins[internalId], scales[internalId], dest, 0);
  }

  /** Retrieves a copy of the quantized vector representation at the specified internal index. */
  public QuantizedVector getQuantizedVector(int internalId) {
    checkInternalBounds(internalId);
    byte[] dataCopy = new byte[dimension];
    System.arraycopy(vectors, internalId * dimension, dataCopy, 0, dimension);
    return new QuantizedVector(dataCopy, mins[internalId], scales[internalId]);
  }

  /**
   * Copies the raw quantized byte values at the specified internal index into the destination
   * array.
   */
  public void copyQuantizedVector(int internalId, byte[] dest) {
    checkInternalBounds(internalId);
    if (dest.length != dimension) {
      throw new IllegalArgumentException(
          "Destination array length ("
              + dest.length
              + ") does not match dimension ("
              + dimension
              + ")");
    }
    System.arraycopy(vectors, internalId * dimension, dest, 0, dimension);
  }

  /**
   * Returns the direct internal contiguous primitive byte buffer.
   *
   * <p><b>Read-only contract:</b> Callers must treat the returned array as read-only.
   */
  public byte[] vectorBuffer() {
    return vectors;
  }

  /**
   * Returns the direct internal array of per-vector minimums.
   *
   * <p><b>Read-only contract:</b> Callers must treat the returned array as read-only.
   */
  public float[] minBuffer() {
    return mins;
  }

  /**
   * Returns the direct internal array of per-vector scales.
   *
   * <p><b>Read-only contract:</b> Callers must treat the returned array as read-only.
   */
  public float[] scaleBuffer() {
    return scales;
  }

  /**
   * Returns the direct internal array of external identifiers indexed by internal ID.
   *
   * <p><b>Read-only contract:</b> Callers must treat the returned array as read-only.
   */
  public long[] externalIdBuffer() {
    return externalIds;
  }

  /** Returns the total number of stored vectors. */
  public int size() {
    return size;
  }

  /** Returns the dimensionality of vectors in this storage. */
  public int dimension() {
    return dimension;
  }

  /** Returns the current allocated capacity. */
  public int capacity() {
    return externalIds.length;
  }

  /** Returns the scalar quantizer used by this storage. */
  public ScalarQuantizer quantizer() {
    return quantizer;
  }

  private void ensureCapacity(int minCapacity) {
    int currentCapacity = externalIds.length;
    if (minCapacity > currentCapacity) {
      int newCapacity = Math.max(minCapacity, currentCapacity * 2);
      vectors = Arrays.copyOf(vectors, newCapacity * dimension);
      mins = Arrays.copyOf(mins, newCapacity);
      scales = Arrays.copyOf(scales, newCapacity);
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
