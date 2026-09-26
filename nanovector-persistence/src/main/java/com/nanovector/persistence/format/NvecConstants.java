package com.nanovector.persistence.format;

import com.nanovector.core.distance.DistanceMetric;
import java.util.Objects;

/** Constants and mappings for the NVEC binary persistence format (v1). */
public final class NvecConstants {

  private NvecConstants() {}

  /** File signature magic bytes ("NVEC" in ASCII). */
  public static final byte[] MAGIC_BYTES = new byte[] {'N', 'V', 'E', 'C'};

  /** Current specification version. */
  public static final short FORMAT_VERSION_1 = 1;

  /** Marker byte indicating Little-Endian encoding (explicit format choice). */
  public static final byte ENDIANNESS_LITTLE = 1;

  /** Binary code for Euclidean distance metric ($L_2^2$). */
  public static final byte METRIC_EUCLIDEAN = 1;

  /** Binary code for Cosine distance metric. */
  public static final byte METRIC_COSINE = 2;

  /** Binary code for Inverted Dot Product distance metric. */
  public static final byte METRIC_DOT_PRODUCT = 3;

  /** Size of the fixed header in bytes. */
  public static final int HEADER_SIZE_BYTES = 32;

  /** Number of reserved alignment bytes in the header. */
  public static final int RESERVED_BYTES_COUNT = 3;

  /** Number of zero-padding bytes in the header. */
  public static final int HEADER_PADDING_BYTES_COUNT = 12;

  /** Size of the metadata length field in bytes. */
  public static final int METADATA_LENGTH_FIELD_BYTES = 4;

  /** Size of FLAT index metadata payload in bytes. */
  public static final int FLAT_METADATA_PAYLOAD_BYTES = 0;

  /** Size of HNSW index metadata payload in bytes. */
  public static final int HNSW_METADATA_PAYLOAD_BYTES = 24;

  /** Size of the CRC32C checksum footer in bytes. */
  public static final int FOOTER_CHECKSUM_SIZE_BYTES = 4;

  /** Minimum size of a valid NVEC file in bytes (header + metadata length + checksum). */
  public static final int MIN_FILE_SIZE_BYTES =
      HEADER_SIZE_BYTES + METADATA_LENGTH_FIELD_BYTES + FOOTER_CHECKSUM_SIZE_BYTES;

  /** Standard file extension for serialized NVEC vector index files. */
  public static final String FILE_EXTENSION = ".nvec";

  /**
   * Converts a {@link DistanceMetric} enum to its 1-byte NVEC format code.
   *
   * @param metric the distance metric
   * @return the corresponding byte code (1, 2, or 3)
   */
  public static byte metricToCode(DistanceMetric metric) {
    Objects.requireNonNull(metric, "Metric must not be null");
    return switch (metric) {
      case EUCLIDEAN -> METRIC_EUCLIDEAN;
      case COSINE -> METRIC_COSINE;
      case DOT_PRODUCT -> METRIC_DOT_PRODUCT;
    };
  }

  /**
   * Converts a 1-byte NVEC format code to its corresponding {@link DistanceMetric} enum.
   *
   * @param code the byte code
   * @return the corresponding {@link DistanceMetric}
   * @throws IllegalArgumentException if the code does not map to any known distance metric
   */
  public static DistanceMetric codeToMetric(byte code) {
    return switch (code) {
      case METRIC_EUCLIDEAN -> DistanceMetric.EUCLIDEAN;
      case METRIC_COSINE -> DistanceMetric.COSINE;
      case METRIC_DOT_PRODUCT -> DistanceMetric.DOT_PRODUCT;
      default -> throw new IllegalArgumentException("Unknown distance metric code: " + code);
    };
  }
}
