package com.nanovector.persistence.reader;

import com.nanovector.core.distance.DistanceCalculator;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.IndexRestorer;
import com.nanovector.core.index.VectorIndex;
import com.nanovector.core.storage.VectorStorage;
import com.nanovector.persistence.exception.CorruptIndexException;
import com.nanovector.persistence.format.HnswMetadata;
import com.nanovector.persistence.format.IndexType;
import com.nanovector.persistence.format.NvecConstants;
import com.nanovector.persistence.format.NvecHeader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;
import java.util.zip.CRC32C;

/**
 * Deserializes vector indexes ({@link FlatIndex} and {@link HnswIndex}) from NVEC v1 binary files.
 *
 * <p>Enforces a strict, defense-in-depth restoration pipeline:
 *
 * <ol>
 *   <li>File size check against format minimums ({@code >= 40} bytes).
 *   <li>Fixed 32-byte header parsing and invariant validation.
 *   <li>Pre-allocation structural sanity checks (overflow detection and minimum size matching).
 *   <li>Hardware-accelerated CRC32C integrity gate over all bytes {@code [0, fileSize - 4)}.
 *   <li>Metadata parsing and forward-compatible extension skipping.
 *   <li>Vector storage buffer and external ID mapping restoration.
 *   <li>Verbatim HNSW topology reconstruction without running edge selection or pruning algorithms.
 * </ol>
 */
public final class NvecReader {

  private static final int BUFFER_SIZE = 64 * 1024; // 64 KB streaming buffer
  private static final int CHUNK_ELEMENTS = 2048; // Max elements per primitive chunk read

  private NvecReader() {}

  /**
   * Reads and restores any supported {@link VectorIndex} from an NVEC v1 file, using SIMD
   * acceleration by default.
   *
   * @param path path to .nvec file
   * @return restored {@link FlatIndex} or {@link HnswIndex}
   * @throws IOException on file I/O errors or if index validation fails
   */
  public static VectorIndex read(Path path) throws IOException {
    return read(path, true);
  }

  /**
   * Reads and restores any supported {@link VectorIndex} from an NVEC v1 file.
   *
   * @param path path to .nvec file
   * @param useSimd whether to use SIMD distance calculator
   * @return restored {@link FlatIndex} or {@link HnswIndex}
   * @throws IOException on file I/O errors or if index validation fails
   */
  public static VectorIndex read(Path path, boolean useSimd) throws IOException {
    Objects.requireNonNull(path, "Path must not be null");

    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      long fileSize = channel.size();

      // 1. File size minimum check
      if (fileSize < NvecConstants.MIN_FILE_SIZE_BYTES) {
        throw new CorruptIndexException(
            "File size "
                + fileSize
                + " bytes is smaller than minimum valid NVEC file size ("
                + NvecConstants.MIN_FILE_SIZE_BYTES
                + " bytes)");
      }

      // 2. Read and validate 32-byte Header
      ByteBuffer headerBuffer = ByteBuffer.allocate(NvecConstants.HEADER_SIZE_BYTES);
      readFully(channel, headerBuffer);
      headerBuffer.flip();
      NvecHeader header = NvecHeader.read(headerBuffer);

      // 3. Pre-allocation structural sanity checks (prevent OOM from corrupt dimensions)
      validateStructuralBounds(header, fileSize);

      // 4. CRC32C integrity gate (over [0, fileSize - 4))
      verifyCrc32c(channel, fileSize);

      // 5. Dispatch restoration based on index type
      if (header.indexType() == IndexType.FLAT) {
        return restoreFlat(channel, header, fileSize, useSimd);
      } else if (header.indexType() == IndexType.HNSW) {
        return restoreHnsw(channel, header, fileSize, useSimd);
      } else {
        throw new CorruptIndexException("Unsupported index type: " + header.indexType());
      }
    }
  }

  /**
   * Reads and restores a {@link FlatIndex} from an NVEC v1 file.
   *
   * @param path path to .nvec file
   * @return restored {@link FlatIndex}
   * @throws IOException on file I/O errors or index format violations
   */
  public static FlatIndex readFlat(Path path) throws IOException {
    return readFlat(path, true);
  }

  /**
   * Reads and restores a {@link FlatIndex} from an NVEC v1 file.
   *
   * @param path path to .nvec file
   * @param useSimd whether to use SIMD distance calculator
   * @return restored {@link FlatIndex}
   * @throws IOException on file I/O errors or index format violations
   */
  public static FlatIndex readFlat(Path path, boolean useSimd) throws IOException {
    VectorIndex index = read(path, useSimd);
    if (!(index instanceof FlatIndex flatIndex)) {
      throw new CorruptIndexException(
          "Expected FLAT index, but found: " + index.getClass().getSimpleName());
    }
    return flatIndex;
  }

  /**
   * Reads and restores an {@link HnswIndex} from an NVEC v1 file.
   *
   * @param path path to .nvec file
   * @return restored {@link HnswIndex}
   * @throws IOException on file I/O errors or index format violations
   */
  public static HnswIndex readHnsw(Path path) throws IOException {
    return readHnsw(path, true);
  }

  /**
   * Reads and restores an {@link HnswIndex} from an NVEC v1 file.
   *
   * @param path path to .nvec file
   * @param useSimd whether to use SIMD distance calculator
   * @return restored {@link HnswIndex}
   * @throws IOException on file I/O errors or index format violations
   */
  public static HnswIndex readHnsw(Path path, boolean useSimd) throws IOException {
    VectorIndex index = read(path, useSimd);
    if (!(index instanceof HnswIndex hnswIndex)) {
      throw new CorruptIndexException(
          "Expected HNSW index, but found: " + index.getClass().getSimpleName());
    }
    return hnswIndex;
  }

  // ── Restoration Pipelines ───────────────────────────────────────────

  private static FlatIndex restoreFlat(
      FileChannel channel, NvecHeader header, long fileSize, boolean useSimd) throws IOException {
    int dimension = header.dimension();
    int size = header.vectorCount();

    // Position immediately after 32-byte header
    channel.position(NvecConstants.HEADER_SIZE_BYTES);

    // Read metadata_length
    ByteBuffer metaLenBuf =
        ByteBuffer.allocate(NvecConstants.METADATA_LENGTH_FIELD_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    readFully(channel, metaLenBuf);
    metaLenBuf.flip();
    int metadataLength = metaLenBuf.getInt();
    if (metadataLength < 0) {
      throw new CorruptIndexException("Negative metadata_length in FLAT index: " + metadataLength);
    }
    if (metadataLength > 0) {
      // Forward-compatible extension skip
      channel.position(channel.position() + metadataLength);
    }

    // Read Vector Storage & External IDs
    VectorStorage storage = readVectorStorage(channel, dimension, size, header.metric());

    // Verify stream reached exactly the CRC32C footer position
    long expectedFooterPos = fileSize - NvecConstants.FOOTER_CHECKSUM_SIZE_BYTES;
    if (channel.position() != expectedFooterPos) {
      throw new CorruptIndexException(
          "Extraneous bytes in FLAT index file: current position "
              + channel.position()
              + ", expected "
              + expectedFooterPos);
    }

    DistanceCalculator calculator = DistanceCalculator.create(header.metric(), useSimd);
    return IndexRestorer.restoreFlat(dimension, header.metric(), storage, calculator);
  }

  private static HnswIndex restoreHnsw(
      FileChannel channel, NvecHeader header, long fileSize, boolean useSimd) throws IOException {
    int dimension = header.dimension();
    int size = header.vectorCount();

    channel.position(NvecConstants.HEADER_SIZE_BYTES);

    // Read metadata_length
    ByteBuffer metaLenBuf =
        ByteBuffer.allocate(NvecConstants.METADATA_LENGTH_FIELD_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    readFully(channel, metaLenBuf);
    metaLenBuf.flip();
    int metadataLength = metaLenBuf.getInt();
    if (metadataLength < NvecConstants.HNSW_METADATA_PAYLOAD_BYTES) {
      throw new CorruptIndexException(
          "Insufficient metadata_length for HNSW index: "
              + metadataLength
              + " bytes (expected at least "
              + NvecConstants.HNSW_METADATA_PAYLOAD_BYTES
              + ")");
    }

    // Read 24-byte HNSW metadata payload
    ByteBuffer metaPayloadBuf =
        ByteBuffer.allocate(NvecConstants.HNSW_METADATA_PAYLOAD_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    readFully(channel, metaPayloadBuf);
    metaPayloadBuf.flip();
    HnswMetadata metadata = HnswMetadata.read(metaPayloadBuf);

    // Skip any unknown extension bytes for forward compatibility
    int extensionBytes = metadataLength - NvecConstants.HNSW_METADATA_PAYLOAD_BYTES;
    if (extensionBytes > 0) {
      channel.position(channel.position() + extensionBytes);
    }

    // Validate metadata consistency with vector count
    if (size == 0) {
      if (metadata.maxLevel() != -1 || metadata.entryPointId() != -1) {
        throw new CorruptIndexException(
            "Empty HNSW index must have maxLevel=-1 and entryPointId=-1, but got maxLevel="
                + metadata.maxLevel()
                + ", entryPointId="
                + metadata.entryPointId());
      }
    } else {
      if (metadata.maxLevel() < 0) {
        throw new CorruptIndexException(
            "Non-empty HNSW index must have maxLevel >= 0, but got: " + metadata.maxLevel());
      }
      if (metadata.entryPointId() < 0 || metadata.entryPointId() >= size) {
        throw new CorruptIndexException(
            "HNSW entryPointId out of bounds: "
                + metadata.entryPointId()
                + " (vector count="
                + size
                + ")");
      }
    }

    // Read Vector Storage & External IDs
    VectorStorage storage = readVectorStorage(channel, dimension, size, header.metric());

    // Build HnswConfig with stored hyperparameters
    HnswConfig config =
        new HnswConfig(
            metadata.m(),
            metadata.m0(),
            metadata.efConstruction(),
            metadata.defaultEfSearch(),
            1.0 / Math.log(metadata.m()),
            null);

    // Reconstruct verbatim graph topology
    HnswGraph graph = readGraphTopology(channel, size, metadata, config);

    // Verify stream reached exactly the CRC32C footer position
    long expectedFooterPos = fileSize - NvecConstants.FOOTER_CHECKSUM_SIZE_BYTES;
    if (channel.position() != expectedFooterPos) {
      throw new CorruptIndexException(
          "Extraneous bytes in HNSW index file: current position "
              + channel.position()
              + ", expected "
              + expectedFooterPos);
    }

    DistanceCalculator calculator = DistanceCalculator.create(header.metric(), useSimd);
    return IndexRestorer.restoreHnsw(
        dimension, header.metric(), config, storage, graph, calculator);
  }

  // ── Block Parsers ───────────────────────────────────────────────────

  private static VectorStorage readVectorStorage(
      FileChannel channel, int dimension, int size, DistanceMetric metric) throws IOException {
    long totalFloatsLong = (long) size * dimension;
    int totalFloats = (int) totalFloatsLong;
    float[] vectors = new float[totalFloats];
    long[] externalIds = new long[size];

    if (size > 0) {
      // 1. Read Vector Buffer
      ByteBuffer byteBuffer =
          ByteBuffer.allocate(CHUNK_ELEMENTS * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

      int floatsRead = 0;
      while (floatsRead < totalFloats) {
        int count = Math.min(CHUNK_ELEMENTS, totalFloats - floatsRead);
        byteBuffer.clear();
        byteBuffer.limit(count * Float.BYTES);
        readFully(channel, byteBuffer);
        byteBuffer.flip();
        floatBuffer.clear();
        floatBuffer.get(vectors, floatsRead, count);
        floatsRead += count;
      }

      // Validate floats are finite and cosine vectors are unit-normalized
      for (int i = 0; i < size; i++) {
        int offset = i * dimension;
        double normSq = 0.0;
        for (int d = 0; d < dimension; d++) {
          float val = vectors[offset + d];
          if (!Float.isFinite(val)) {
            throw new CorruptIndexException(
                "Non-finite float value (NaN or Inf) detected in vector " + i + ": " + val);
          }
          if (metric == DistanceMetric.COSINE) {
            normSq += (double) val * val;
          }
        }
        if (metric == DistanceMetric.COSINE) {
          if (Math.abs(normSq - 1.0) > 1e-3) {
            throw new CorruptIndexException(
                "Vector "
                    + i
                    + " is not unit-normalized for COSINE metric (norm^2 = "
                    + normSq
                    + ")");
          }
        }
      }

      // 2. Read External IDs
      ByteBuffer idByteBuffer =
          ByteBuffer.allocate(CHUNK_ELEMENTS * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      LongBuffer longBuffer = idByteBuffer.asLongBuffer();

      int idsRead = 0;
      while (idsRead < size) {
        int count = Math.min(CHUNK_ELEMENTS, size - idsRead);
        idByteBuffer.clear();
        idByteBuffer.limit(count * Long.BYTES);
        readFully(channel, idByteBuffer);
        idByteBuffer.flip();
        longBuffer.clear();
        longBuffer.get(externalIds, idsRead, count);
        idsRead += count;
      }
    }

    try {
      return new VectorStorage(dimension, size, vectors, externalIds);
    } catch (IllegalArgumentException e) {
      throw new CorruptIndexException("Failed to restore VectorStorage: " + e.getMessage(), e);
    }
  }

  private static HnswGraph readGraphTopology(
      FileChannel channel, int size, HnswMetadata metadata, HnswConfig config) throws IOException {
    HnswGraph graph = new HnswGraph(config);
    if (size == 0) {
      return graph;
    }

    int graphMaxLevel = metadata.maxLevel();
    int m0 = metadata.m0();
    int m = metadata.m();

    BufferedChannelReader reader = new BufferedChannelReader(channel, BUFFER_SIZE);

    for (int i = 0; i < size; i++) {
      int nodeMaxLevel = Byte.toUnsignedInt(reader.readByte());
      if (nodeMaxLevel > graphMaxLevel) {
        throw new CorruptIndexException(
            "Node " + i + " maxLevel " + nodeMaxLevel + " exceeds graph maxLevel " + graphMaxLevel);
      }

      HnswNode node = new HnswNode(i, nodeMaxLevel);

      for (int l = 0; l <= nodeMaxLevel; l++) {
        int degree = Short.toUnsignedInt(reader.readShort());
        int maxDegree = (l == 0) ? m0 : m;
        if (degree > maxDegree) {
          throw new CorruptIndexException(
              "Node "
                  + i
                  + " layer "
                  + l
                  + " degree "
                  + degree
                  + " exceeds maximum allowed "
                  + maxDegree);
        }

        int[] neighbors = new int[degree];
        reader.readInts(neighbors, 0, degree);

        // Validate neighbor IDs and ensure no self-loops or duplicates
        for (int a = 0; a < degree; a++) {
          int nbrA = neighbors[a];
          if (nbrA < 0 || nbrA >= size) {
            throw new CorruptIndexException(
                "Neighbor ID out of bounds: " + nbrA + " on node " + i + " layer " + l);
          }
          if (nbrA == i) {
            throw new CorruptIndexException(
                "Self-loop detected: node " + i + " is neighbor to itself on layer " + l);
          }
          for (int b = a + 1; b < degree; b++) {
            if (nbrA == neighbors[b]) {
              throw new CorruptIndexException(
                  "Duplicate neighbor ID " + nbrA + " on node " + i + " layer " + l);
            }
          }
        }

        node.setNeighbors(l, neighbors);
      }

      graph.addNode(node);
    }

    graph.setEntryPoint(metadata.entryPointId(), graphMaxLevel);

    // Synchronize underlying channel position after buffered reading
    reader.syncChannelPosition();
    return graph;
  }

  // ── Structural and Integrity Validation ─────────────────────────────

  private static void validateStructuralBounds(NvecHeader header, long fileSize)
      throws CorruptIndexException {
    int dimension = header.dimension();
    int vectorCount = header.vectorCount();

    // Check for integer overflow in element count calculation
    long totalFloats = (long) vectorCount * dimension;
    if (totalFloats < 0 || totalFloats > (Integer.MAX_VALUE - 8)) {
      throw new CorruptIndexException(
          "Structural dimension * vectorCount overflow: " + totalFloats + " floats");
    }

    long vectorBytes = totalFloats * Float.BYTES;
    long externalIdBytes = (long) vectorCount * Long.BYTES;

    if (header.indexType() == IndexType.FLAT) {
      long minFlatSize =
          NvecConstants.HEADER_SIZE_BYTES
              + NvecConstants.METADATA_LENGTH_FIELD_BYTES
              + vectorBytes
              + externalIdBytes
              + NvecConstants.FOOTER_CHECKSUM_SIZE_BYTES;
      if (fileSize < minFlatSize) {
        throw new CorruptIndexException(
            "FLAT index file size smaller than structural minimum: expected at least "
                + minFlatSize
                + " bytes, but found "
                + fileSize
                + " bytes");
      }
    } else if (header.indexType() == IndexType.HNSW) {
      long minHnswSize =
          NvecConstants.HEADER_SIZE_BYTES
              + NvecConstants.METADATA_LENGTH_FIELD_BYTES
              + NvecConstants.HNSW_METADATA_PAYLOAD_BYTES
              + vectorBytes
              + externalIdBytes
              + NvecConstants.FOOTER_CHECKSUM_SIZE_BYTES;
      if (fileSize < minHnswSize) {
        throw new CorruptIndexException(
            "HNSW index file size is smaller than structural minimum: expected at least "
                + minHnswSize
                + " bytes, but found "
                + fileSize
                + " bytes");
      }
    }
  }

  private static void verifyCrc32c(FileChannel channel, long fileSize) throws IOException {
    long payloadLength = fileSize - NvecConstants.FOOTER_CHECKSUM_SIZE_BYTES;
    CRC32C crc = new CRC32C();
    ByteBuffer chunk = ByteBuffer.allocate(BUFFER_SIZE);

    channel.position(0);
    long remaining = payloadLength;
    while (remaining > 0) {
      chunk.clear();
      int toRead = (int) Math.min(chunk.capacity(), remaining);
      chunk.limit(toRead);
      int bytesRead = channel.read(chunk);
      if (bytesRead < 0) {
        throw new CorruptIndexException("Unexpected EOF while streaming CRC32C verification");
      }
      crc.update(chunk.array(), 0, bytesRead);
      remaining -= bytesRead;
    }

    // Read stored uint32 CRC footer
    ByteBuffer footerBuf =
        ByteBuffer.allocate(NvecConstants.FOOTER_CHECKSUM_SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    channel.position(payloadLength);
    readFully(channel, footerBuf);
    footerBuf.flip();
    long storedCrc = Integer.toUnsignedLong(footerBuf.getInt());
    long computedCrc = crc.getValue();

    if (storedCrc != computedCrc) {
      throw new CorruptIndexException(
          "CRC32C checksum mismatch: stored "
              + Long.toHexString(storedCrc)
              + ", computed "
              + Long.toHexString(computedCrc));
    }
  }

  private static void readFully(FileChannel channel, ByteBuffer buf) throws IOException {
    while (buf.hasRemaining()) {
      int read = channel.read(buf);
      if (read < 0) {
        throw new CorruptIndexException("Unexpected end of file while reading binary data");
      }
    }
  }

  // ── Buffered Channel Reader ─────────────────────────────────────────

  private static final class BufferedChannelReader {
    private final FileChannel channel;
    private final ByteBuffer buffer;
    private long channelOffset;

    BufferedChannelReader(FileChannel channel, int capacity) throws IOException {
      this.channel = channel;
      this.channelOffset = channel.position();
      this.buffer = ByteBuffer.allocate(capacity).order(ByteOrder.LITTLE_ENDIAN);
      this.buffer.limit(0); // initially empty
    }

    byte readByte() throws IOException {
      ensureBytes(1);
      return buffer.get();
    }

    short readShort() throws IOException {
      ensureBytes(2);
      return buffer.getShort();
    }

    void readInts(int[] dst, int offset, int length) throws IOException {
      int remaining = length;
      int curOffset = offset;
      while (remaining > 0) {
        if (!buffer.hasRemaining()) {
          fillBuffer();
        }
        int availableInts = buffer.remaining() / 4;
        if (availableInts == 0) {
          buffer.compact();
          int read = channel.read(buffer);
          if (read < 0) {
            throw new CorruptIndexException("Unexpected EOF while reading integers");
          }
          buffer.flip();
          availableInts = buffer.remaining() / 4;
          if (availableInts == 0) {
            throw new CorruptIndexException("Unexpected EOF while reading integers");
          }
        }
        int toRead = Math.min(remaining, availableInts);
        for (int k = 0; k < toRead; k++) {
          dst[curOffset++] = buffer.getInt();
        }
        remaining -= toRead;
      }
    }

    private void ensureBytes(int count) throws IOException {
      if (buffer.remaining() < count) {
        buffer.compact();
        while (buffer.position() < count) {
          int read = channel.read(buffer);
          if (read < 0) {
            throw new CorruptIndexException("Unexpected EOF: required " + count + " bytes");
          }
        }
        buffer.flip();
      }
    }

    private void fillBuffer() throws IOException {
      buffer.clear();
      int read = channel.read(buffer);
      if (read < 0) {
        throw new CorruptIndexException("Unexpected EOF while reading buffer");
      }
      buffer.flip();
    }

    void syncChannelPosition() throws IOException {
      long unconsumed = buffer.remaining();
      channel.position(channel.position() - unconsumed);
    }
  }
}
