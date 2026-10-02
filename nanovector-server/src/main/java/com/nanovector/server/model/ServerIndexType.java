package com.nanovector.server.model;

import java.util.Locale;

/** Enumeration of vector index types supported by NanoVector Server. */
public enum ServerIndexType {
  /** Exact flat scan on-heap FP32 index. */
  FLAT,

  /** Hierarchical Navigable Small World on-heap FP32 approximate nearest-neighbor index. */
  HNSW,

  /** 8-bit scalar quantized (SQ8) on-heap approximate nearest-neighbor index. */
  HNSW_SQ8,

  /** 8-bit scalar quantized (SQ8) native off-heap (FFM) approximate nearest-neighbor index. */
  HNSW_SQ8_OFFHEAP;

  /** Returns true if this index type can be serialized/deserialized by NVEC v1 persistence. */
  public boolean isPersistenceSupported() {
    return this == FLAT || this == HNSW;
  }

  /**
   * Parses an index type string case-insensitively.
   *
   * @param typeStr input string
   * @return matching ServerIndexType
   * @throws IllegalArgumentException if the type is unknown or blank
   */
  public static ServerIndexType parse(String typeStr) {
    if (typeStr == null || typeStr.isBlank()) {
      throw new IllegalArgumentException("Index type must not be null or blank");
    }
    String normalized = typeStr.trim().toUpperCase(Locale.ROOT);
    try {
      return ServerIndexType.valueOf(normalized);
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException(
          "Unsupported index type '"
              + typeStr
              + "'. Supported types: FLAT, HNSW, HNSW_SQ8, HNSW_SQ8_OFFHEAP");
    }
  }
}
