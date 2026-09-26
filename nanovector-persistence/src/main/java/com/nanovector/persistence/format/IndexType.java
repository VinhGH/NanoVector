package com.nanovector.persistence.format;

/** Enumeration of index types supported by the NVEC binary format. */
public enum IndexType {
  /** Exact flat scan vector index. */
  FLAT((byte) 1),

  /** Hierarchical Navigable Small World approximate nearest-neighbor index. */
  HNSW((byte) 2);

  private final byte code;

  IndexType(byte code) {
    this.code = code;
  }

  /** Returns the 1-byte binary code representing this index type. */
  public byte code() {
    return code;
  }

  /**
   * Resolves the {@link IndexType} corresponding to the provided byte code.
   *
   * @param code the binary code (1 for FLAT, 2 for HNSW)
   * @return the corresponding {@link IndexType}
   * @throws IllegalArgumentException if the code does not map to any known index type
   */
  public static IndexType fromCode(byte code) {
    return switch (code) {
      case 1 -> FLAT;
      case 2 -> HNSW;
      default -> throw new IllegalArgumentException("Unknown index type code: " + code);
    };
  }
}
