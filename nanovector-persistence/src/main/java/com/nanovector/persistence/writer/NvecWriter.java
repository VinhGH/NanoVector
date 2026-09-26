package com.nanovector.persistence.writer;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.VectorIndex;
import com.nanovector.core.storage.VectorDataView;
import com.nanovector.persistence.format.HnswMetadata;
import com.nanovector.persistence.format.IndexType;
import com.nanovector.persistence.format.NvecConstants;
import com.nanovector.persistence.format.NvecHeader;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Objects;
import java.util.zip.CRC32C;

/**
 * Serializes vector indexes ({@link FlatIndex} and {@link HnswIndex}) into the NVEC v1 binary
 * format with streaming hardware-accelerated CRC32C corruption detection and atomic file
 * replacement.
 *
 * <p><b>Persistence boundaries:</b> {@code NvecWriter} is strictly a serialization engine. It only
 * serializes the current state of an index without modifying index state, running HNSW
 * construction/pruning algorithms, or mutating stored vectors.
 */
public final class NvecWriter {

  private static final int BUFFER_SIZE = 64 * 1024; // 64 KB streaming buffer
  private static final int CHUNK_ELEMENTS = 2048; // Elements per primitive chunk transfer

  private NvecWriter() {}

  /**
   * Serializes any supported {@link VectorIndex} to the specified target path.
   *
   * @param index the vector index to serialize
   * @param targetPath destination file path (.nvec)
   * @throws IOException on file I/O errors
   * @throws IllegalArgumentException if index contains invalid floats or unnormalized cosine
   *     vectors
   */
  public static void write(VectorIndex index, Path targetPath) throws IOException {
    Objects.requireNonNull(index, "VectorIndex must not be null");
    Objects.requireNonNull(targetPath, "Target path must not be null");

    if (index instanceof FlatIndex flatIndex) {
      write(flatIndex, targetPath);
    } else if (index instanceof HnswIndex hnswIndex) {
      write(hnswIndex, targetPath);
    } else {
      throw new IllegalArgumentException(
          "Unsupported VectorIndex implementation: " + index.getClass().getName());
    }
  }

  /**
   * Serializes a {@link FlatIndex} to the specified target path in NVEC v1 format.
   *
   * @param index the flat index to serialize
   * @param targetPath destination file path (.nvec)
   * @throws IOException on file I/O errors
   */
  public static void write(FlatIndex index, Path targetPath) throws IOException {
    Objects.requireNonNull(index, "FlatIndex must not be null");
    Objects.requireNonNull(targetPath, "Target path must not be null");

    VectorDataView view = index.vectorData();
    validateVectors(view, index.metric());

    NvecHeader header =
        NvecHeader.of(IndexType.FLAT, index.metric(), index.dimension(), index.size());

    executeAtomicWrite(
        targetPath,
        (crcOut, bufOut) -> {
          // 1. Fixed Header (32 bytes)
          writeHeader(header, crcOut);

          // 2. Metadata Block (FLAT metadata_length = 0)
          writeFlatMetadata(crcOut);

          // 3. Vector Storage Block
          writeVectorStorage(view, crcOut);

          // 4. External ID Mapping Block
          writeExternalIds(view, crcOut);
        });
  }

  /**
   * Serializes an {@link HnswIndex} to the specified target path in NVEC v1 format, preserving
   * graph topology verbatim without recomputing edges or running pruning heuristics.
   *
   * @param index the HNSW index to serialize
   * @param targetPath destination file path (.nvec)
   * @throws IOException on file I/O errors
   */
  public static void write(HnswIndex index, Path targetPath) throws IOException {
    Objects.requireNonNull(index, "HnswIndex must not be null");
    Objects.requireNonNull(targetPath, "Target path must not be null");

    VectorDataView view = index.vectorData();
    validateVectors(view, index.metric());

    HnswGraph graph = index.graph();
    validateGraphTopology(graph, index.config(), index.size());

    NvecHeader header =
        NvecHeader.of(IndexType.HNSW, index.metric(), index.dimension(), index.size());

    HnswConfig config = index.config();
    int maxLevel = graph.maxLevel();
    int entryPointId = graph.entryPointId();
    HnswMetadata metadata =
        new HnswMetadata(
            config.m(),
            config.m0(),
            config.efConstruction(),
            config.efSearch(),
            maxLevel,
            entryPointId);

    executeAtomicWrite(
        targetPath,
        (crcOut, bufOut) -> {
          // 1. Fixed Header (32 bytes)
          writeHeader(header, crcOut);

          // 2. Metadata Block (metadata_length = 24 + 24 bytes payload)
          writeHnswMetadata(metadata, crcOut);

          // 3. Vector Storage Block
          writeVectorStorage(view, crcOut);

          // 4. External ID Mapping Block
          writeExternalIds(view, crcOut);

          // 5. Graph Topology Block
          writeGraphTopology(graph, index.size(), config, crcOut);
        });
  }

  // ── Block Writers ───────────────────────────────────────────────────

  private static void writeHeader(NvecHeader header, OutputStream out) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(NvecConstants.HEADER_SIZE_BYTES);
    header.write(buffer);
    out.write(buffer.array(), 0, NvecConstants.HEADER_SIZE_BYTES);
  }

  private static void writeFlatMetadata(OutputStream out) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(NvecConstants.METADATA_LENGTH_FIELD_BYTES);
    buffer.order(ByteOrder.LITTLE_ENDIAN);
    buffer.putInt(NvecConstants.FLAT_METADATA_PAYLOAD_BYTES); // 0
    out.write(buffer.array(), 0, NvecConstants.METADATA_LENGTH_FIELD_BYTES);
  }

  private static void writeHnswMetadata(HnswMetadata metadata, OutputStream out)
      throws IOException {
    int totalMetaBytes =
        NvecConstants.METADATA_LENGTH_FIELD_BYTES + NvecConstants.HNSW_METADATA_PAYLOAD_BYTES;
    ByteBuffer buffer = ByteBuffer.allocate(totalMetaBytes);
    buffer.order(ByteOrder.LITTLE_ENDIAN);
    buffer.putInt(NvecConstants.HNSW_METADATA_PAYLOAD_BYTES); // 24
    metadata.write(buffer);
    out.write(buffer.array(), 0, totalMetaBytes);
  }

  private static void writeVectorStorage(VectorDataView view, OutputStream out) throws IOException {
    int size = view.size();
    int dimension = view.dimension();
    int totalFloats = size * dimension;
    if (totalFloats == 0) {
      return;
    }

    float[] vectors = view.vectorBuffer();
    ByteBuffer byteBuffer =
        ByteBuffer.allocate(CHUNK_ELEMENTS * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

    int offset = 0;
    while (offset < totalFloats) {
      int count = Math.min(CHUNK_ELEMENTS, totalFloats - offset);
      floatBuffer.clear();
      floatBuffer.put(vectors, offset, count);
      out.write(byteBuffer.array(), 0, count * Float.BYTES);
      offset += count;
    }
  }

  private static void writeExternalIds(VectorDataView view, OutputStream out) throws IOException {
    int size = view.size();
    if (size == 0) {
      return;
    }

    long[] externalIds = view.externalIdBuffer();
    ByteBuffer byteBuffer =
        ByteBuffer.allocate(CHUNK_ELEMENTS * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    LongBuffer longBuffer = byteBuffer.asLongBuffer();

    int offset = 0;
    while (offset < size) {
      int count = Math.min(CHUNK_ELEMENTS, size - offset);
      longBuffer.clear();
      longBuffer.put(externalIds, offset, count);
      out.write(byteBuffer.array(), 0, count * Long.BYTES);
      offset += count;
    }
  }

  private static void writeGraphTopology(
      HnswGraph graph, int size, HnswConfig config, OutputStream out) throws IOException {
    if (size == 0) {
      return;
    }

    int m0 = config.m0();
    int m = config.m();

    // Reusable byte buffer for packing node topology
    ByteBuffer nodeBuffer = ByteBuffer.allocate(BUFFER_SIZE).order(ByteOrder.LITTLE_ENDIAN);

    for (int i = 0; i < size; i++) {
      HnswNode node = graph.getNode(i);
      int nodeMaxLevel = node.maxLevel();

      // Flush buffer if insufficient space for at least 1 byte
      if (nodeBuffer.remaining() < 1) {
        out.write(nodeBuffer.array(), 0, nodeBuffer.position());
        nodeBuffer.clear();
      }
      nodeBuffer.put((byte) nodeMaxLevel);

      for (int l = 0; l <= nodeMaxLevel; l++) {
        int degree = node.degree(l);
        int[] neighbors = node.getNeighbors(l);
        int bytesNeeded = 2 + degree * 4;

        if (nodeBuffer.remaining() < bytesNeeded) {
          out.write(nodeBuffer.array(), 0, nodeBuffer.position());
          nodeBuffer.clear();
        }

        // If a single layer's degree exceeds buffer size, flush and write directly
        if (bytesNeeded > nodeBuffer.capacity()) {
          ByteBuffer largeBuffer = ByteBuffer.allocate(bytesNeeded).order(ByteOrder.LITTLE_ENDIAN);
          largeBuffer.putShort((short) degree);
          for (int nbr : neighbors) {
            largeBuffer.putInt(nbr);
          }
          out.write(largeBuffer.array(), 0, bytesNeeded);
        } else {
          nodeBuffer.putShort((short) degree);
          for (int nbr : neighbors) {
            nodeBuffer.putInt(nbr);
          }
        }
      }
    }

    if (nodeBuffer.position() > 0) {
      out.write(nodeBuffer.array(), 0, nodeBuffer.position());
      nodeBuffer.clear();
    }
  }

  // ── Validation Helpers ──────────────────────────────────────────────

  private static void validateVectors(VectorDataView view, DistanceMetric metric) {
    int size = view.size();
    int dim = view.dimension();
    float[] buffer = view.vectorBuffer();

    for (int i = 0; i < size; i++) {
      int offset = i * dim;
      double normSq = 0.0;
      for (int d = 0; d < dim; d++) {
        float val = buffer[offset + d];
        if (!Float.isFinite(val)) {
          throw new IllegalArgumentException(
              "Vector at internal ID "
                  + i
                  + " contains non-finite float value (NaN or Inf): "
                  + val);
        }
        if (metric == DistanceMetric.COSINE) {
          normSq += (double) val * val;
        }
      }
      if (metric == DistanceMetric.COSINE && size > 0) {
        if (Math.abs(normSq - 1.0) > 1e-3) {
          throw new IllegalArgumentException(
              "Vector at internal ID "
                  + i
                  + " is not unit-normalized for COSINE metric (norm^2 = "
                  + normSq
                  + ")");
        }
      }
    }
  }

  private static void validateGraphTopology(HnswGraph graph, HnswConfig config, int size) {
    if (size == 0) {
      if (graph.size() != 0 || graph.maxLevel() != -1 || graph.entryPointId() != -1) {
        throw new IllegalStateException("Empty index must have empty graph structure");
      }
      return;
    }

    int graphMaxLevel = graph.maxLevel();
    int entryPoint = graph.entryPointId();
    if (entryPoint < 0 || entryPoint >= size) {
      throw new IllegalStateException(
          "Invalid graph entry point ID: " + entryPoint + " for graph of size " + size);
    }
    if (graphMaxLevel < 0) {
      throw new IllegalStateException("Non-empty graph must have maxLevel >= 0: " + graphMaxLevel);
    }

    int m0 = config.m0();
    int m = config.m();

    for (int i = 0; i < size; i++) {
      HnswNode node = graph.getNode(i);
      if (node == null || node.internalId() != i) {
        throw new IllegalStateException("Corrupt node mapping at internal ID " + i);
      }
      int nodeMaxLevel = node.maxLevel();
      if (nodeMaxLevel < 0 || nodeMaxLevel > graphMaxLevel) {
        throw new IllegalStateException(
            "Node " + i + " maxLevel " + nodeMaxLevel + " exceeds graph maxLevel " + graphMaxLevel);
      }

      for (int l = 0; l <= nodeMaxLevel; l++) {
        int degree = node.degree(l);
        int maxAllowed = (l == 0) ? m0 : m;
        if (degree > maxAllowed) {
          throw new IllegalStateException(
              "Node "
                  + i
                  + " layer "
                  + l
                  + " degree "
                  + degree
                  + " exceeds maximum allowed "
                  + maxAllowed);
        }
        int[] neighbors = node.getNeighbors(l);
        for (int a = 0; a < degree; a++) {
          int nbrA = neighbors[a];
          if (nbrA < 0 || nbrA >= size) {
            throw new IllegalStateException(
                "Neighbor ID out of bounds: " + nbrA + " on node " + i + " layer " + l);
          }
          if (nbrA == i) {
            throw new IllegalStateException(
                "Self-loop detected: node " + i + " is neighbor to itself on layer " + l);
          }
          for (int b = a + 1; b < degree; b++) {
            if (nbrA == neighbors[b]) {
              throw new IllegalStateException(
                  "Duplicate neighbor ID " + nbrA + " on node " + i + " layer " + l);
            }
          }
        }
      }
    }
  }

  // ── Atomic Write Pipeline ───────────────────────────────────────────

  @FunctionalInterface
  private interface BlockWriter {
    void write(OutputStream crcOut, OutputStream bufOut) throws IOException;
  }

  private static void executeAtomicWrite(Path targetPath, BlockWriter writer) throws IOException {
    Path parentDir = targetPath.toAbsolutePath().getParent();
    if (parentDir != null) {
      Files.createDirectories(parentDir);
    }

    String fileName = targetPath.getFileName().toString();
    Path tempPath = Files.createTempFile(parentDir, fileName + "-", ".tmp");
    boolean success = false;

    try {
      try (OutputStream fileOut = Files.newOutputStream(tempPath);
          BufferedOutputStream bufOut = new BufferedOutputStream(fileOut, BUFFER_SIZE)) {

        CRC32C crc = new CRC32C();
        CrcOutputStream crcOut = new CrcOutputStream(bufOut, crc);

        // Execute block serialization passing through CRC32C
        writer.write(crcOut, bufOut);

        // Flush all payload bytes into CRC32C
        crcOut.flush();

        // 6. Checksum Footer: Write 4-byte CRC32C Little-Endian directly to bufOut
        // (WITHOUT passing through crcOut, so footer is excluded from its own checksum)
        long checksum = crc.getValue();
        writeUint32LittleEndian(bufOut, checksum);
        bufOut.flush();
      }

      // Atomic move with fallback for filesystems that do not support ATOMIC_MOVE
      try {
        Files.move(
            tempPath,
            targetPath,
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING);
      } catch (AtomicMoveNotSupportedException | UnsupportedOperationException e) {
        Files.move(tempPath, targetPath, StandardCopyOption.REPLACE_EXISTING);
      }

      success = true;
    } finally {
      if (!success) {
        try {
          Files.deleteIfExists(tempPath);
        } catch (IOException ignored) {
          // Best effort cleanup
        }
      }
    }
  }

  private static void writeUint32LittleEndian(OutputStream out, long value) throws IOException {
    out.write((int) (value & 0xFF));
    out.write((int) ((value >>> 8) & 0xFF));
    out.write((int) ((value >>> 16) & 0xFF));
    out.write((int) ((value >>> 24) & 0xFF));
  }

  /**
   * Direct streaming wrapper that feeds every byte to CRC32C before writing to downstream stream.
   */
  private static final class CrcOutputStream extends OutputStream {
    private final OutputStream out;
    private final CRC32C crc;

    CrcOutputStream(OutputStream out, CRC32C crc) {
      this.out = out;
      this.crc = crc;
    }

    @Override
    public void write(int b) throws IOException {
      out.write(b);
      crc.update((byte) b);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
      out.write(b, off, len);
      crc.update(b, off, len);
    }

    @Override
    public void flush() throws IOException {
      out.flush();
    }
  }
}
