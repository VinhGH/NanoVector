package com.nanovector.core.offheap;

import com.nanovector.core.quantization.AsymmetricSq8Quantizer;
import com.nanovector.core.quantization.QuantizedVector;
import com.nanovector.core.quantization.ScalarQuantizer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;

/**
 * Off-heap contiguous primitive storage for 8-bit scalar quantized vectors and affine metadata.
 *
 * <p>Stores quantized vectors and per-vector affine parameters ($v_{\min}$ and $\text{scale}$) in
 * an interleaved native memory layout allocated via Foreign Function &amp; Memory (FFM) {@link
 * Arena}:
 *
 * <pre>
 *   [float v_min (4B)][float scale (4B)][byte[dimension] quantizedCoordinates]
 * </pre>
 *
 * <p>Stride per vector is 8-byte aligned:
 *
 * <pre>
 *   stride = ((8 + dimension + 7) / 8) * 8
 * </pre>
 *
 * <p>For $D=128$, each vector record consumes exactly 136 bytes ($8 + 128 = 136$, already a
 * multiple of 8). This interleaved representation ensures that affine metadata ($v_{\min},
 * \text{scale}$) and vector coordinates share the same cache line / adjacent cache lines,
 * maximizing L1/L2 cache locality during SIMD distance evaluation.
 */
public final class OffHeapQuantizedVectorStorage implements AutoCloseable {

  private static final int DEFAULT_INITIAL_CAPACITY = 1024;

  private final int dimension;
  private final ScalarQuantizer quantizer;
  private final Arena arena;
  private final boolean ownsArena;
  private final long vectorStrideBytes;

  private int capacity;
  private int size;
  private MemorySegment vectorSegment;
  private MemorySegment externalIdSegment;
  private final Map<Long, Integer> externalToInternal;
  private boolean isClosed;

  public OffHeapQuantizedVectorStorage(int dimension) {
    this(dimension, DEFAULT_INITIAL_CAPACITY, AsymmetricSq8Quantizer.INSTANCE);
  }

  public OffHeapQuantizedVectorStorage(int dimension, int initialCapacity) {
    this(dimension, initialCapacity, AsymmetricSq8Quantizer.INSTANCE);
  }

  public OffHeapQuantizedVectorStorage(
      int dimension, int initialCapacity, ScalarQuantizer quantizer) {
    this(dimension, initialCapacity, quantizer, Arena.ofConfined(), true);
  }

  public OffHeapQuantizedVectorStorage(
      int dimension, int initialCapacity, ScalarQuantizer quantizer, Arena arena) {
    this(dimension, initialCapacity, quantizer, arena, false);
  }

  private OffHeapQuantizedVectorStorage(
      int dimension,
      int initialCapacity,
      ScalarQuantizer quantizer,
      Arena arena,
      boolean ownsArena) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (initialCapacity <= 0) {
      throw new IllegalArgumentException("Initial capacity must be positive: " + initialCapacity);
    }
    this.dimension = dimension;
    this.quantizer = Objects.requireNonNull(quantizer, "Quantizer must not be null");
    this.arena = Objects.requireNonNull(arena, "Arena must not be null");
    this.ownsArena = ownsArena;

    this.capacity = initialCapacity;
    this.size = 0;

    long unpaddedStride = 8L + dimension;
    this.vectorStrideBytes = ((unpaddedStride + 7L) / 8L) * 8L;

    this.vectorSegment = arena.allocate((long) capacity * vectorStrideBytes, 8);
    this.externalIdSegment = arena.allocate((long) capacity * Long.BYTES, 8);
    this.externalToInternal = HashMap.newHashMap(initialCapacity);
    this.isClosed = false;
  }

  /**
   * Quantizes and appends an FP32 vector into off-heap storage.
   *
   * @param externalId unique user-defined external ID
   * @param vector full-precision FP32 vector
   * @return internal index assigned to the newly stored vector
   */
  public int add(long externalId, float[] vector) {
    checkClosed();
    Objects.requireNonNull(vector, "Vector must not be null");
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Vector dimension " + vector.length + " does not match storage dimension " + dimension);
    }
    if (externalToInternal.containsKey(externalId)) {
      throw new IllegalArgumentException("External ID " + externalId + " already exists");
    }

    ensureCapacity(size + 1);

    QuantizedVector q = quantizer.quantize(vector);

    long offset = (long) size * vectorStrideBytes;
    vectorSegment.set(ValueLayout.JAVA_FLOAT, offset, q.min());
    vectorSegment.set(ValueLayout.JAVA_FLOAT, offset + 4L, q.scale());
    MemorySegment.copy(q.data(), 0, vectorSegment, ValueLayout.JAVA_BYTE, offset + 8L, dimension);

    externalIdSegment.set(ValueLayout.JAVA_LONG, (long) size * Long.BYTES, externalId);
    externalToInternal.put(externalId, size);

    return size++;
  }

  /** Returns per-vector affine minimum parameter ($v_{\min}$). */
  public float getMin(int internalId) {
    checkClosed();
    checkBounds(internalId);
    return vectorSegment.get(ValueLayout.JAVA_FLOAT, (long) internalId * vectorStrideBytes);
  }

  /** Returns per-vector affine scaling step size parameter ($\text{scale}$). */
  public float getScale(int internalId) {
    checkClosed();
    checkBounds(internalId);
    return vectorSegment.get(ValueLayout.JAVA_FLOAT, (long) internalId * vectorStrideBytes + 4L);
  }

  /**
   * Returns byte offset in {@link #vectorSegment()} where the vector's quantized coordinates begin.
   */
  public long getVectorOffset(int internalId) {
    checkClosed();
    checkBounds(internalId);
    return (long) internalId * vectorStrideBytes + 8L;
  }

  /** Returns the underlying contiguous native memory segment storing vector records. */
  public MemorySegment vectorSegment() {
    checkClosed();
    return vectorSegment;
  }

  /** Returns a copy of the quantized byte coordinates for the specified internal ID. */
  public byte[] getQuantizedVector(int internalId) {
    byte[] dest = new byte[dimension];
    getQuantizedVector(internalId, dest);
    return dest;
  }

  /** Copies the quantized byte coordinates for the specified internal ID into {@code dest}. */
  public void getQuantizedVector(int internalId, byte[] dest) {
    checkClosed();
    checkBounds(internalId);
    Objects.requireNonNull(dest, "Destination buffer must not be null");
    if (dest.length < dimension) {
      throw new IllegalArgumentException(
          "Destination array length " + dest.length + " is smaller than dimension " + dimension);
    }
    long offset = (long) internalId * vectorStrideBytes + 8L;
    MemorySegment.copy(vectorSegment, ValueLayout.JAVA_BYTE, offset, dest, 0, dimension);
  }

  /** Dequantizes and reconstructs the full-precision FP32 vector for the specified internal ID. */
  public float[] getVector(int internalId) {
    checkClosed();
    checkBounds(internalId);
    long offset = (long) internalId * vectorStrideBytes;
    float min = vectorSegment.get(ValueLayout.JAVA_FLOAT, offset);
    float scale = vectorSegment.get(ValueLayout.JAVA_FLOAT, offset + 4L);
    byte[] raw = new byte[dimension];
    MemorySegment.copy(vectorSegment, ValueLayout.JAVA_BYTE, offset + 8L, raw, 0, dimension);
    return quantizer.dequantize(new QuantizedVector(raw, min, scale));
  }

  /** Returns the external ID associated with the specified internal index. */
  public long getExternalId(int internalId) {
    checkClosed();
    checkBounds(internalId);
    return externalIdSegment.get(ValueLayout.JAVA_LONG, (long) internalId * Long.BYTES);
  }

  /** Returns the internal index associated with the specified external ID. */
  public int getInternalId(long externalId) {
    checkClosed();
    Integer id = externalToInternal.get(externalId);
    if (id == null) {
      throw new NoSuchElementException("External ID not found: " + externalId);
    }
    return id;
  }

  public boolean contains(long externalId) {
    checkClosed();
    return externalToInternal.containsKey(externalId);
  }

  private void ensureCapacity(int requiredCapacity) {
    if (requiredCapacity <= capacity) {
      return;
    }
    int newCap = Math.max(capacity * 2, requiredCapacity);

    MemorySegment newVectorSegment = arena.allocate((long) newCap * vectorStrideBytes, 8);
    MemorySegment.copy(vectorSegment, 0L, newVectorSegment, 0L, (long) size * vectorStrideBytes);
    this.vectorSegment = newVectorSegment;

    MemorySegment newExternalIdSegment = arena.allocate((long) newCap * Long.BYTES, 8);
    MemorySegment.copy(externalIdSegment, 0L, newExternalIdSegment, 0L, (long) size * Long.BYTES);
    this.externalIdSegment = newExternalIdSegment;

    this.capacity = newCap;
  }

  private void checkBounds(int internalId) {
    if (internalId < 0 || internalId >= size) {
      throw new IndexOutOfBoundsException(
          "Internal ID " + internalId + " out of bounds, size is " + size);
    }
  }

  private void checkClosed() {
    if (isClosed) {
      throw new IllegalStateException("OffHeapQuantizedVectorStorage has been closed");
    }
  }

  public int size() {
    return size;
  }

  public int capacity() {
    return capacity;
  }

  public int dimension() {
    return dimension;
  }

  public long vectorStrideBytes() {
    return vectorStrideBytes;
  }

  public ScalarQuantizer quantizer() {
    return quantizer;
  }

  /** Total native memory allocated across vector and external ID segments (bytes). */
  public long nativeAllocatedBytes() {
    return vectorSegment.byteSize() + externalIdSegment.byteSize();
  }

  @Override
  public void close() {
    if (!isClosed) {
      isClosed = true;
      if (ownsArena && arena.scope().isAlive()) {
        arena.close();
      }
    }
  }
}
