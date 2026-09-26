package com.nanovector.persistence.format;

import com.nanovector.persistence.exception.CorruptIndexException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Immutable metadata record for HNSW vector index persistence (24 bytes).
 *
 * @param m maximum outgoing edges per node at layers &gt; 0 (must be &ge; 2)
 * @param m0 maximum outgoing edges per node at layer 0 (must be &ge; m)
 * @param efConstruction dynamic candidate list size during graph construction (must be &gt; 0)
 * @param defaultEfSearch default search candidate list size (must be &gt; 0)
 * @param maxLevel maximum layer index present in the graph (-1 if empty, &ge; 0 if non-empty)
 * @param entryPointId internal ID of the entry point node (-1 if empty, &ge; 0 if non-empty)
 */
public record HnswMetadata(
    int m, int m0, int efConstruction, int defaultEfSearch, int maxLevel, int entryPointId) {

  public HnswMetadata {
    if (m < 2) {
      throw new IllegalArgumentException("m must be at least 2, but got: " + m);
    }
    if (m0 < m) {
      throw new IllegalArgumentException("m0 must be at least m (" + m + "), but got: " + m0);
    }
    if (efConstruction <= 0) {
      throw new IllegalArgumentException(
          "efConstruction must be positive, but got: " + efConstruction);
    }
    if (defaultEfSearch <= 0) {
      throw new IllegalArgumentException(
          "defaultEfSearch must be positive, but got: " + defaultEfSearch);
    }
    if (maxLevel < -1) {
      throw new IllegalArgumentException("maxLevel must be >= -1, but got: " + maxLevel);
    }
    if (entryPointId < -1) {
      throw new IllegalArgumentException("entryPointId must be >= -1, but got: " + entryPointId);
    }
    if ((maxLevel == -1 && entryPointId != -1) || (maxLevel != -1 && entryPointId == -1)) {
      throw new IllegalArgumentException(
          "maxLevel and entryPointId must both be -1 for empty graph, or both non-negative");
    }
  }

  /**
   * Serializes the 24-byte HNSW metadata block into the provided buffer in Little-Endian byte
   * order.
   *
   * @param buffer target byte buffer with at least 24 bytes remaining
   */
  public void write(ByteBuffer buffer) {
    Objects.requireNonNull(buffer, "Buffer must not be null");
    ByteOrder originalOrder = buffer.order();
    buffer.order(ByteOrder.LITTLE_ENDIAN);
    try {
      buffer.putInt(m);
      buffer.putInt(m0);
      buffer.putInt(efConstruction);
      buffer.putInt(defaultEfSearch);
      buffer.putInt(maxLevel);
      buffer.putInt(entryPointId);
    } finally {
      buffer.order(originalOrder);
    }
  }

  /**
   * Deserializes an {@link HnswMetadata} instance from the provided buffer in Little-Endian byte
   * order.
   *
   * @param buffer source byte buffer with at least 24 bytes remaining
   * @return deserialized and validated {@link HnswMetadata}
   * @throws CorruptIndexException if buffer has insufficient bytes or values violate invariants
   */
  public static HnswMetadata read(ByteBuffer buffer) throws CorruptIndexException {
    Objects.requireNonNull(buffer, "Buffer must not be null");
    if (buffer.remaining() < NvecConstants.HNSW_METADATA_PAYLOAD_BYTES) {
      throw new CorruptIndexException(
          "Insufficient buffer size for HNSW metadata: required "
              + NvecConstants.HNSW_METADATA_PAYLOAD_BYTES
              + " bytes, but only "
              + buffer.remaining()
              + " available");
    }
    ByteOrder originalOrder = buffer.order();
    buffer.order(ByteOrder.LITTLE_ENDIAN);
    try {
      int m = buffer.getInt();
      int m0 = buffer.getInt();
      int efConstruction = buffer.getInt();
      int defaultEfSearch = buffer.getInt();
      int maxLevel = buffer.getInt();
      int entryPointId = buffer.getInt();
      try {
        return new HnswMetadata(m, m0, efConstruction, defaultEfSearch, maxLevel, entryPointId);
      } catch (IllegalArgumentException e) {
        throw new CorruptIndexException("Invalid HNSW metadata values: " + e.getMessage(), e);
      }
    } finally {
      buffer.order(originalOrder);
    }
  }
}
