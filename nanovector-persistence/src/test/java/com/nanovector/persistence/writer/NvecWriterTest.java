package com.nanovector.persistence.writer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.VectorIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.storage.VectorDataView;
import com.nanovector.persistence.format.HnswMetadata;
import com.nanovector.persistence.format.IndexType;
import com.nanovector.persistence.format.NvecConstants;
import com.nanovector.persistence.format.NvecHeader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.CRC32C;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NvecWriterTest {

  @TempDir Path tempDir;

  @Test
  @DisplayName("Should write FLAT index with exact byte offsets, payload, and CRC32C footer")
  void testFlatIndexWriteSuccess() throws IOException {
    int dim = 4;
    FlatIndex index = new FlatIndex(dim, DistanceMetric.EUCLIDEAN);
    index.insert(101L, new float[] {1.0f, 2.0f, 3.0f, 4.0f});
    index.insert(102L, new float[] {5.0f, 6.0f, 7.0f, 8.0f});
    index.insert(103L, new float[] {9.0f, 10.0f, 11.0f, 12.0f});

    Path targetFile = tempDir.resolve("flat_test.nvec");
    NvecWriter.write(index, targetFile);

    assertThat(Files.exists(targetFile)).isTrue();

    // Exact formula check: 40 + N * (4D + 8) = 40 + 3 * (16 + 8) = 40 + 72 = 112 bytes
    long fileSize = Files.size(targetFile);
    assertThat(fileSize).isEqualTo(112L);

    byte[] allBytes = Files.readAllBytes(targetFile);
    ByteBuffer buffer = ByteBuffer.wrap(allBytes).order(ByteOrder.LITTLE_ENDIAN);

    // 1. Header (32 bytes)
    NvecHeader header = NvecHeader.read(buffer);
    assertThat(header.version()).isEqualTo(NvecConstants.FORMAT_VERSION_1);
    assertThat(header.endianness()).isEqualTo(NvecConstants.ENDIANNESS_LITTLE);
    assertThat(header.indexType()).isEqualTo(IndexType.FLAT);
    assertThat(header.metric()).isEqualTo(DistanceMetric.EUCLIDEAN);
    assertThat(header.dimension()).isEqualTo(dim);
    assertThat(header.vectorCount()).isEqualTo(3);

    // 2. Metadata length field (4 bytes)
    int metaLen = buffer.getInt();
    assertThat(metaLen).isEqualTo(0);

    // 3. Vector Storage (3 * 4 * 4 = 48 bytes)
    for (int expected = 1; expected <= 12; expected++) {
      assertThat(buffer.getFloat()).isEqualTo((float) expected);
    }

    // 4. External IDs (3 * 8 = 24 bytes)
    assertThat(buffer.getLong()).isEqualTo(101L);
    assertThat(buffer.getLong()).isEqualTo(102L);
    assertThat(buffer.getLong()).isEqualTo(103L);

    // 5. CRC32C Footer (last 4 bytes)
    assertThat(buffer.position()).isEqualTo(108);
    int storedCrcInt = buffer.getInt();
    long storedCrc = Integer.toUnsignedLong(storedCrcInt);

    CRC32C crc = new CRC32C();
    crc.update(allBytes, 0, 108);
    assertThat(storedCrc).isEqualTo(crc.getValue());
  }

  @Test
  @DisplayName("Should write empty FLAT index (N=0) with exactly 40 bytes")
  void testEmptyFlatIndex() throws IOException {
    FlatIndex emptyIndex = new FlatIndex(128, DistanceMetric.COSINE);
    Path targetFile = tempDir.resolve("empty_flat.nvec");

    NvecWriter.write(emptyIndex, targetFile);

    assertThat(Files.size(targetFile)).isEqualTo(40L);

    byte[] allBytes = Files.readAllBytes(targetFile);
    ByteBuffer buffer = ByteBuffer.wrap(allBytes).order(ByteOrder.LITTLE_ENDIAN);

    NvecHeader header = NvecHeader.read(buffer);
    assertThat(header.indexType()).isEqualTo(IndexType.FLAT);
    assertThat(header.metric()).isEqualTo(DistanceMetric.COSINE);
    assertThat(header.dimension()).isEqualTo(128);
    assertThat(header.vectorCount()).isEqualTo(0);

    int metaLen = buffer.getInt();
    assertThat(metaLen).isEqualTo(0);

    int storedCrcInt = buffer.getInt();
    CRC32C crc = new CRC32C();
    crc.update(allBytes, 0, 36);
    assertThat(Integer.toUnsignedLong(storedCrcInt)).isEqualTo(crc.getValue());
  }

  @Test
  @DisplayName("Should write HNSW index with exact metadata, verbatim topology, and CRC32C footer")
  void testHnswIndexWriteSuccess() throws IOException {
    int dim = 3;
    HnswConfig config = HnswConfig.withSeed(12345L).withEfSearch(40);
    HnswIndex index = new HnswIndex(dim, DistanceMetric.EUCLIDEAN, config);

    index.insert(1L, new float[] {0.1f, 0.2f, 0.3f});
    index.insert(2L, new float[] {0.4f, 0.5f, 0.6f});
    index.insert(3L, new float[] {0.7f, 0.8f, 0.9f});
    index.insert(4L, new float[] {1.0f, 1.1f, 1.2f});

    Path targetFile = tempDir.resolve("hnsw_test.nvec");
    NvecWriter.write(index, targetFile);

    byte[] allBytes = Files.readAllBytes(targetFile);
    ByteBuffer buffer = ByteBuffer.wrap(allBytes).order(ByteOrder.LITTLE_ENDIAN);

    // 1. Header (32 bytes)
    NvecHeader header = NvecHeader.read(buffer);
    assertThat(header.indexType()).isEqualTo(IndexType.HNSW);
    assertThat(header.metric()).isEqualTo(DistanceMetric.EUCLIDEAN);
    assertThat(header.dimension()).isEqualTo(dim);
    assertThat(header.vectorCount()).isEqualTo(4);

    // 2. Metadata (4B length + 24B payload = 28 bytes)
    int metaLen = buffer.getInt();
    assertThat(metaLen).isEqualTo(24);
    HnswMetadata meta = HnswMetadata.read(buffer);
    assertThat(meta.m()).isEqualTo(config.m());
    assertThat(meta.m0()).isEqualTo(config.m0());
    assertThat(meta.efConstruction()).isEqualTo(config.efConstruction());
    assertThat(meta.defaultEfSearch()).isEqualTo(40);
    assertThat(meta.maxLevel()).isEqualTo(index.graph().maxLevel());
    assertThat(meta.entryPointId()).isEqualTo(index.graph().entryPointId());

    // 3. Vector Storage (4 * 3 * 4 = 48 bytes)
    for (int i = 0; i < 4; i++) {
      float[] stored = index.vectorData().vectorBuffer();
      for (int d = 0; d < dim; d++) {
        assertThat(buffer.getFloat()).isEqualTo(stored[i * dim + d]);
      }
    }

    // 4. External IDs (4 * 8 = 32 bytes)
    assertThat(buffer.getLong()).isEqualTo(1L);
    assertThat(buffer.getLong()).isEqualTo(2L);
    assertThat(buffer.getLong()).isEqualTo(3L);
    assertThat(buffer.getLong()).isEqualTo(4L);

    // 5. Topology Block (verbatim from graph)
    HnswGraph graph = index.graph();
    for (int i = 0; i < 4; i++) {
      HnswNode node = graph.getNode(i);
      int nodeMaxLevel = Byte.toUnsignedInt(buffer.get());
      assertThat(nodeMaxLevel).isEqualTo(node.maxLevel());

      for (int l = 0; l <= nodeMaxLevel; l++) {
        int degree = Short.toUnsignedInt(buffer.getShort());
        assertThat(degree).isEqualTo(node.degree(l));
        int[] expectedNeighbors = node.getNeighbors(l);
        for (int exp : expectedNeighbors) {
          assertThat(buffer.getInt()).isEqualTo(exp);
        }
      }
    }

    // 6. Checksum footer
    int payloadEnd = allBytes.length - 4;
    assertThat(buffer.position()).isEqualTo(payloadEnd);
    int storedCrcInt = buffer.getInt();
    CRC32C crc = new CRC32C();
    crc.update(allBytes, 0, payloadEnd);
    assertThat(Integer.toUnsignedLong(storedCrcInt)).isEqualTo(crc.getValue());
  }

  @Test
  @DisplayName("Should write empty HNSW index (N=0) with exactly 64 bytes")
  void testEmptyHnswIndex() throws IOException {
    HnswIndex emptyIndex =
        new HnswIndex(64, DistanceMetric.DOT_PRODUCT, HnswConfig.defaultConfig());
    Path targetFile = tempDir.resolve("empty_hnsw.nvec");

    NvecWriter.write(emptyIndex, targetFile);

    // Header (32) + meta length (4) + meta payload (24) + CRC (4) = 64 bytes
    assertThat(Files.size(targetFile)).isEqualTo(64L);

    byte[] allBytes = Files.readAllBytes(targetFile);
    ByteBuffer buffer = ByteBuffer.wrap(allBytes).order(ByteOrder.LITTLE_ENDIAN);

    NvecHeader header = NvecHeader.read(buffer);
    assertThat(header.indexType()).isEqualTo(IndexType.HNSW);
    assertThat(header.vectorCount()).isEqualTo(0);

    int metaLen = buffer.getInt();
    assertThat(metaLen).isEqualTo(24);

    HnswMetadata meta = HnswMetadata.read(buffer);
    assertThat(meta.maxLevel()).isEqualTo(-1);
    assertThat(meta.entryPointId()).isEqualTo(-1);

    int storedCrcInt = buffer.getInt();
    CRC32C crc = new CRC32C();
    crc.update(allBytes, 0, 60);
    assertThat(Integer.toUnsignedLong(storedCrcInt)).isEqualTo(crc.getValue());
  }

  @Test
  @DisplayName("Should atomically replace existing target file")
  void testReplaceExistingTargetFile() throws IOException {
    Path targetFile = tempDir.resolve("replace_test.nvec");

    FlatIndex index1 = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    index1.insert(1L, new float[] {1.0f, 1.0f});
    NvecWriter.write(index1, targetFile);
    long size1 = Files.size(targetFile);

    FlatIndex index2 = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    index2.insert(1L, new float[] {1.0f, 1.0f});
    index2.insert(2L, new float[] {2.0f, 2.0f});
    index2.insert(3L, new float[] {3.0f, 3.0f});
    NvecWriter.write(index2, targetFile);
    long size2 = Files.size(targetFile);

    assertThat(size2).isGreaterThan(size1);

    byte[] bytes = Files.readAllBytes(targetFile);
    NvecHeader header = NvecHeader.read(ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN));
    assertThat(header.vectorCount()).isEqualTo(3);
  }

  @Test
  @DisplayName("Should reject vectors with NaN or Infinite floats without leaving .tmp files")
  void testRejectNonFiniteFloats() throws IOException {
    FlatIndex index = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    index.insert(1L, new float[] {1.0f, 2.0f});

    // Directly corrupt raw buffer to simulate corrupted memory before write
    index.vectorData().vectorBuffer()[0] = Float.NaN;

    Path targetFile = tempDir.resolve("corrupt_nan.nvec");
    assertThatThrownBy(() -> NvecWriter.write(index, targetFile))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("non-finite float");

    assertThat(Files.exists(targetFile)).isFalse();

    // Verify cleanup of any .tmp files
    try (Stream<Path> files = Files.list(tempDir)) {
      assertThat(files.filter(p -> p.toString().endsWith(".tmp"))).isEmpty();
    }
  }

  @Test
  @DisplayName("Should reject unnormalized vector for COSINE metric")
  void testRejectUnnormalizedCosineVector() throws IOException {
    FlatIndex index = new FlatIndex(2, DistanceMetric.COSINE);
    // Insert normalizes, so modify buffer directly afterwards to simulate unnormalized vector
    index.insert(1L, new float[] {1.0f, 0.0f});
    index.vectorData().vectorBuffer()[0] = 5.0f; // norm = 5.0 != 1.0

    Path targetFile = tempDir.resolve("unnormalized_cosine.nvec");
    assertThatThrownBy(() -> NvecWriter.write(index, targetFile))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("not unit-normalized for COSINE metric");

    assertThat(Files.exists(targetFile)).isFalse();

    try (Stream<Path> files = Files.list(tempDir)) {
      assertThat(files.filter(p -> p.toString().endsWith(".tmp"))).isEmpty();
    }
  }

  @Test
  @DisplayName("Should write via generic VectorIndex interface")
  void testGenericVectorIndexWrite() throws IOException {
    VectorIndex flat = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    flat.insert(1L, new float[] {0.5f, 0.5f});
    Path path1 = tempDir.resolve("generic_flat.nvec");
    NvecWriter.write(flat, path1);
    assertThat(Files.exists(path1)).isTrue();

    VectorIndex hnsw = new HnswIndex(2, DistanceMetric.EUCLIDEAN, HnswConfig.defaultConfig());
    hnsw.insert(1L, new float[] {0.5f, 0.5f});
    Path path2 = tempDir.resolve("generic_hnsw.nvec");
    NvecWriter.write(hnsw, path2);
    assertThat(Files.exists(path2)).isTrue();

    // Unsupported VectorIndex custom implementation
    VectorIndex customIndex =
        new VectorIndex() {
          @Override
          public void insert(long id, float[] vector) {}

          @Override
          public List<SearchResult> searchKnn(float[] query, int k) {
            return List.of();
          }

          @Override
          public int size() {
            return 0;
          }

          @Override
          public int dimension() {
            return 2;
          }

          @Override
          public DistanceMetric metric() {
            return DistanceMetric.EUCLIDEAN;
          }

          @Override
          public VectorDataView vectorData() {
            return null;
          }
        };

    Path path3 = tempDir.resolve("custom.nvec");
    assertThatThrownBy(() -> NvecWriter.write(customIndex, path3))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Unsupported VectorIndex implementation");
  }

  @Test
  @DisplayName(
      "Should reject HNSW graph with invalid topology or self-loop without leaving .tmp files")
  void testRejectCorruptedGraphTopology() throws IOException {
    HnswConfig config = HnswConfig.defaultConfig();
    HnswIndex index = new HnswIndex(2, DistanceMetric.EUCLIDEAN, config);
    index.insert(1L, new float[] {0.1f, 0.2f});
    index.insert(2L, new float[] {0.3f, 0.4f});

    // Artificially corrupt topology by injecting a self-loop on node 0
    HnswNode node0 = index.graph().getNode(0);
    node0.addNeighbor(0, 0); // Self-loop!

    Path targetFile = tempDir.resolve("corrupt_topology.nvec");
    assertThatThrownBy(() -> NvecWriter.write(index, targetFile))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("Self-loop detected");

    assertThat(Files.exists(targetFile)).isFalse();

    try (Stream<Path> files = Files.list(tempDir)) {
      assertThat(files.filter(p -> p.toString().endsWith(".tmp"))).isEmpty();
    }
  }

  @Test
  @DisplayName("Should create parent directories if target path parent does not exist")
  void testCreateParentDirectories() throws IOException {
    Path nestedTarget = tempDir.resolve("subdir1").resolve("subdir2").resolve("nested.nvec");
    FlatIndex index = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    index.insert(1L, new float[] {1.0f, 2.0f});

    NvecWriter.write(index, nestedTarget);
    assertThat(Files.exists(nestedTarget)).isTrue();
  }
}
