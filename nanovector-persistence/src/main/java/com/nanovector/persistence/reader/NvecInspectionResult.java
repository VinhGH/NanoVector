package com.nanovector.persistence.reader;

import com.nanovector.persistence.format.HnswMetadata;
import com.nanovector.persistence.format.NvecHeader;
import java.util.Objects;

/**
 * Inspection report detailing the binary header, format parameters, HNSW topology metadata, and
 * hardware-verified CRC32C integrity status of an NVEC v1 binary index file.
 *
 * @param fileSizeBytes total file size on disk in bytes
 * @param header parsed and validated 32-byte NVEC header
 * @param hnswMetadata parsed HNSW metadata block, or {@code null} if index type is FLAT
 * @param metadataLengthBytes raw metadata payload length in bytes
 * @param storedCrc32c uint32 CRC32C checksum stored in the file footer
 * @param computedCrc32c uint32 CRC32C checksum computed over bytes [0, fileSize - 4)
 * @param crcValid whether the computed CRC32C matches the stored CRC32C
 */
public record NvecInspectionResult(
    long fileSizeBytes,
    NvecHeader header,
    HnswMetadata hnswMetadata,
    int metadataLengthBytes,
    long storedCrc32c,
    long computedCrc32c,
    boolean crcValid) {

  public NvecInspectionResult {
    Objects.requireNonNull(header, "Header must not be null");
  }

  /**
   * Returns {@code true} if this inspection result represents an HNSW index.
   *
   * @return {@code true} if HNSW metadata is present
   */
  public boolean isHnsw() {
    return hnswMetadata != null;
  }
}
