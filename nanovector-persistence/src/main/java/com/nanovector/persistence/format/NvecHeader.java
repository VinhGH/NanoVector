package com.nanovector.persistence.format;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.persistence.exception.CorruptIndexException;
import com.nanovector.persistence.exception.UnsupportedVersionException;
import java.io.IOException;
import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Validated 32-byte header representation for the NVEC binary format (v1).
 *
 * @param version format version (must be 1 in v1)
 * @param endianness endianness indicator (must be 1 for Little-Endian)
 * @param indexType type of vector index (FLAT or HNSW)
 * @param metric distance metric used for similarity calculation
 * @param dimension vector dimensionality (must be &gt; 0)
 * @param vectorCount number of stored vectors (must be &ge; 0)
 */
public record NvecHeader(
    short version,
    byte endianness,
    IndexType indexType,
    DistanceMetric metric,
    int dimension,
    int vectorCount) {

  public NvecHeader {
    if (version != NvecConstants.FORMAT_VERSION_1) {
      throw new IllegalArgumentException(
          "Unsupported NVEC version: "
              + version
              + " (supported: "
              + NvecConstants.FORMAT_VERSION_1
              + ")");
    }
    if (endianness != NvecConstants.ENDIANNESS_LITTLE) {
      throw new IllegalArgumentException(
          "Unsupported endianness marker: "
              + endianness
              + " (expected: "
              + NvecConstants.ENDIANNESS_LITTLE
              + ")");
    }
    Objects.requireNonNull(indexType, "IndexType must not be null");
    Objects.requireNonNull(metric, "DistanceMetric must not be null");
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive, got: " + dimension);
    }
    if (vectorCount < 0) {
      throw new IllegalArgumentException("Vector count must be non-negative, got: " + vectorCount);
    }
  }

  /**
   * Convenience factory to create a header with default version (1) and Little-Endian format.
   *
   * @param indexType type of index
   * @param metric distance metric
   * @param dimension vector dimension
   * @param vectorCount number of vectors
   * @return new validated {@link NvecHeader}
   */
  public static NvecHeader of(
      IndexType indexType, DistanceMetric metric, int dimension, int vectorCount) {
    return new NvecHeader(
        NvecConstants.FORMAT_VERSION_1,
        NvecConstants.ENDIANNESS_LITTLE,
        indexType,
        metric,
        dimension,
        vectorCount);
  }

  /**
   * Serializes the 32-byte header into the provided {@link ByteBuffer} in Little-Endian format.
   *
   * @param buffer target byte buffer with at least 32 bytes remaining
   * @throws BufferOverflowException if buffer has fewer than 32 bytes remaining
   */
  public void write(ByteBuffer buffer) {
    Objects.requireNonNull(buffer, "Buffer must not be null");
    if (buffer.remaining() < NvecConstants.HEADER_SIZE_BYTES) {
      throw new BufferOverflowException();
    }
    ByteOrder originalOrder = buffer.order();
    buffer.order(ByteOrder.LITTLE_ENDIAN);
    try {
      buffer.put(NvecConstants.MAGIC_BYTES);
      buffer.putShort(version);
      buffer.put(endianness);
      buffer.put(indexType.code());
      buffer.put(NvecConstants.metricToCode(metric));
      buffer.put((byte) 0);
      buffer.put((byte) 0);
      buffer.put((byte) 0);
      buffer.putInt(dimension);
      buffer.putInt(vectorCount);
      for (int i = 0; i < NvecConstants.HEADER_PADDING_BYTES_COUNT; i++) {
        buffer.put((byte) 0);
      }
    } finally {
      buffer.order(originalOrder);
    }
  }

  /**
   * Reads, parses, and validates a 32-byte header from the provided {@link ByteBuffer} in
   * Little-Endian format.
   *
   * @param buffer source byte buffer containing at least 32 bytes
   * @return parsed and validated {@link NvecHeader}
   * @throws CorruptIndexException if header invariants are violated or buffer has corrupted bytes
   * @throws UnsupportedVersionException if the version field is not supported
   * @throws IOException on general I/O or buffer errors
   */
  public static NvecHeader read(ByteBuffer buffer) throws IOException {
    Objects.requireNonNull(buffer, "Buffer must not be null");
    if (buffer.remaining() < NvecConstants.HEADER_SIZE_BYTES) {
      throw new CorruptIndexException(
          "Insufficient buffer size to read NVEC header: required "
              + NvecConstants.HEADER_SIZE_BYTES
              + " bytes, but only "
              + buffer.remaining()
              + " available");
    }
    ByteOrder originalOrder = buffer.order();
    buffer.order(ByteOrder.LITTLE_ENDIAN);
    try {
      byte[] magic = new byte[4];
      buffer.get(magic);
      if (!Arrays.equals(magic, NvecConstants.MAGIC_BYTES)) {
        throw new CorruptIndexException(
            "Invalid NVEC magic bytes: expected 'NVEC', got " + Arrays.toString(magic));
      }

      short version = buffer.getShort();
      if (version != NvecConstants.FORMAT_VERSION_1) {
        throw new UnsupportedVersionException(
            "Unsupported NVEC format version: "
                + version
                + " (supported: "
                + NvecConstants.FORMAT_VERSION_1
                + ")");
      }

      byte endianness = buffer.get();
      if (endianness != NvecConstants.ENDIANNESS_LITTLE) {
        throw new CorruptIndexException(
            "Unsupported endianness byte: "
                + endianness
                + " (expected Little-Endian="
                + NvecConstants.ENDIANNESS_LITTLE
                + ")");
      }

      byte indexTypeCode = buffer.get();
      IndexType indexType;
      try {
        indexType = IndexType.fromCode(indexTypeCode);
      } catch (IllegalArgumentException e) {
        throw new CorruptIndexException("Invalid index type code: " + indexTypeCode, e);
      }

      byte metricCode = buffer.get();
      DistanceMetric metric;
      try {
        metric = NvecConstants.codeToMetric(metricCode);
      } catch (IllegalArgumentException e) {
        throw new CorruptIndexException("Invalid distance metric code: " + metricCode, e);
      }

      for (int i = 0; i < NvecConstants.RESERVED_BYTES_COUNT; i++) {
        byte r = buffer.get();
        if (r != 0) {
          throw new CorruptIndexException("Non-zero reserved byte at index " + i + ": " + r);
        }
      }

      int dimension = buffer.getInt();
      if (dimension <= 0) {
        throw new CorruptIndexException("Dimension must be positive, got: " + dimension);
      }

      int vectorCount = buffer.getInt();
      if (vectorCount < 0) {
        throw new CorruptIndexException("Vector count must be non-negative, got: " + vectorCount);
      }

      for (int i = 0; i < NvecConstants.HEADER_PADDING_BYTES_COUNT; i++) {
        byte p = buffer.get();
        if (p != 0) {
          throw new CorruptIndexException("Non-zero header padding byte at index " + i + ": " + p);
        }
      }

      return new NvecHeader(version, endianness, indexType, metric, dimension, vectorCount);
    } finally {
      buffer.order(originalOrder);
    }
  }
}
