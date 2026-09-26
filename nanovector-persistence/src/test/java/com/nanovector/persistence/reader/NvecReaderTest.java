package com.nanovector.persistence.reader;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.persistence.exception.CorruptIndexException;
import com.nanovector.persistence.exception.UnsupportedVersionException;
import com.nanovector.persistence.writer.NvecWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.zip.CRC32C;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NvecReaderTest {

  @TempDir Path tempDir;

  // ═════════════════════════════════════════════════════════════════════
  // TIER 1 — BYTE CORRECTNESS (ROUND-TRIP)
  // ═════════════════════════════════════════════════════════════════════

  @Test
  @DisplayName("Should successfully round-trip FLAT index with exact byte and state preservation")
  void testFlatIndexRoundTrip() throws IOException {
    int dim = 4;
    FlatIndex original = new FlatIndex(dim, DistanceMetric.EUCLIDEAN);
    original.insert(101L, new float[] {1.5f, 2.5f, 3.5f, 4.5f});
    original.insert(102L, new float[] {5.5f, 6.5f, 7.5f, 8.5f});
    original.insert(103L, new float[] {9.5f, 10.5f, 11.5f, 12.5f});

    Path file = tempDir.resolve("flat_roundtrip.nvec");
    NvecWriter.write(original, file);

    FlatIndex restored = NvecReader.readFlat(file);

    assertThat(restored.dimension()).isEqualTo(original.dimension());
    assertThat(restored.size()).isEqualTo(original.size());
    assertThat(restored.metric()).isEqualTo(original.metric());

    float[] origBuf = original.vectorData().vectorBuffer();
    float[] restBuf = restored.vectorData().vectorBuffer();
    for (int i = 0; i < original.size() * dim; i++) {
      assertThat(restBuf[i]).isEqualTo(origBuf[i]);
    }

    for (int i = 0; i < original.size(); i++) {
      assertThat(restored.vectorData().getExternalId(i))
          .isEqualTo(original.vectorData().getExternalId(i));
    }
  }

  @Test
  @DisplayName("Should successfully round-trip empty FLAT index (N=0)")
  void testEmptyFlatIndexRoundTrip() throws IOException {
    FlatIndex original = new FlatIndex(16, DistanceMetric.DOT_PRODUCT);
    Path file = tempDir.resolve("empty_flat.nvec");
    NvecWriter.write(original, file);

    FlatIndex restored = NvecReader.readFlat(file);
    assertThat(restored.size()).isEqualTo(0);
    assertThat(restored.dimension()).isEqualTo(16);
    assertThat(restored.metric()).isEqualTo(DistanceMetric.DOT_PRODUCT);
  }

  @Test
  @DisplayName(
      "Should successfully round-trip HNSW index with verbatim graph topology preservation")
  void testHnswIndexRoundTrip() throws IOException {
    int dim = 8;
    HnswConfig config = HnswConfig.withSeed(999L).withEfSearch(75).withEfConstruction(150);
    HnswIndex original = new HnswIndex(dim, DistanceMetric.EUCLIDEAN, config);

    Random rng = new Random(42);
    for (long id = 1; id <= 25; id++) {
      float[] vec = new float[dim];
      for (int d = 0; d < dim; d++) {
        vec[d] = rng.nextFloat();
      }
      original.insert(id, vec);
    }

    Path file = tempDir.resolve("hnsw_roundtrip.nvec");
    NvecWriter.write(original, file);

    HnswIndex restored = NvecReader.readHnsw(file);

    // Verify index properties
    assertThat(restored.dimension()).isEqualTo(original.dimension());
    assertThat(restored.size()).isEqualTo(original.size());
    assertThat(restored.metric()).isEqualTo(original.metric());

    // Verify hyperparameters
    assertThat(restored.config().m()).isEqualTo(original.config().m());
    assertThat(restored.config().m0()).isEqualTo(original.config().m0());
    assertThat(restored.config().efConstruction()).isEqualTo(original.config().efConstruction());
    assertThat(restored.config().efSearch()).isEqualTo(75);

    // Verify vectors & IDs
    float[] origBuf = original.vectorData().vectorBuffer();
    float[] restBuf = restored.vectorData().vectorBuffer();
    for (int i = 0; i < original.size() * dim; i++) {
      assertThat(restBuf[i]).isEqualTo(origBuf[i]);
    }
    for (int i = 0; i < original.size(); i++) {
      assertThat(restored.vectorData().getExternalId(i))
          .isEqualTo(original.vectorData().getExternalId(i));
    }

    // Verify graph structure verbatim
    HnswGraph origGraph = original.graph();
    HnswGraph restGraph = restored.graph();
    assertThat(restGraph.size()).isEqualTo(origGraph.size());
    assertThat(restGraph.maxLevel()).isEqualTo(origGraph.maxLevel());
    assertThat(restGraph.entryPointId()).isEqualTo(origGraph.entryPointId());

    for (int i = 0; i < original.size(); i++) {
      HnswNode origNode = origGraph.getNode(i);
      HnswNode restNode = restGraph.getNode(i);

      assertThat(restNode.internalId()).isEqualTo(origNode.internalId());
      assertThat(restNode.maxLevel()).isEqualTo(origNode.maxLevel());

      for (int l = 0; l <= origNode.maxLevel(); l++) {
        assertThat(restNode.degree(l)).isEqualTo(origNode.degree(l));
        assertThat(restNode.getNeighbors(l)).containsExactly(origNode.getNeighbors(l));
      }
    }
  }

  @Test
  @DisplayName("Should successfully round-trip empty HNSW index (N=0)")
  void testEmptyHnswIndexRoundTrip() throws IOException {
    HnswIndex original = new HnswIndex(32, DistanceMetric.COSINE, HnswConfig.defaultConfig());
    Path file = tempDir.resolve("empty_hnsw.nvec");
    NvecWriter.write(original, file);

    HnswIndex restored = NvecReader.readHnsw(file);
    assertThat(restored.size()).isEqualTo(0);
    assertThat(restored.dimension()).isEqualTo(32);
    assertThat(restored.graph().maxLevel()).isEqualTo(-1);
    assertThat(restored.graph().entryPointId()).isEqualTo(-1);
  }

  // ═════════════════════════════════════════════════════════════════════
  // TIER 2 — BEHAVIORAL EQUIVALENCE
  // ═════════════════════════════════════════════════════════════════════

  @Test
  @DisplayName("Restored FLAT index must produce identical search results and ranking order")
  void testFlatBehavioralEquivalence() throws IOException {
    int dim = 16;
    FlatIndex original = new FlatIndex(dim, DistanceMetric.COSINE);
    Random rng = new Random(101);

    for (int i = 1; i <= 50; i++) {
      float[] vec = new float[dim];
      for (int d = 0; d < dim; d++) {
        vec[d] = rng.nextFloat() - 0.5f;
      }
      original.insert(i, vec);
    }

    Path file = tempDir.resolve("flat_behavior.nvec");
    NvecWriter.write(original, file);
    FlatIndex restored = NvecReader.readFlat(file);

    // Query multiple random vectors
    for (int q = 0; q < 10; q++) {
      float[] query = new float[dim];
      for (int d = 0; d < dim; d++) {
        query[d] = rng.nextFloat() - 0.5f;
      }

      List<SearchResult> origRes = original.searchKnn(query, 5);
      List<SearchResult> restRes = restored.searchKnn(query, 5);

      assertThat(restRes).hasSameSizeAs(origRes);
      for (int k = 0; k < origRes.size(); k++) {
        SearchResult o = origRes.get(k);
        SearchResult r = restRes.get(k);
        assertThat(r.id()).isEqualTo(o.id());
        assertThat(r.distance())
            .isCloseTo(o.distance(), org.assertj.core.data.Offset.offset(1e-5f));
      }
    }
  }

  @Test
  @DisplayName("Restored HNSW index must produce identical search results and ranking order")
  void testHnswBehavioralEquivalence() throws IOException {
    int dim = 16;
    HnswIndex original =
        new HnswIndex(dim, DistanceMetric.EUCLIDEAN, HnswConfig.withSeed(777L).withEfSearch(50));
    Random rng = new Random(202);

    for (int i = 1; i <= 100; i++) {
      float[] vec = new float[dim];
      for (int d = 0; d < dim; d++) {
        vec[d] = rng.nextFloat();
      }
      original.insert(i * 10L, vec);
    }

    Path file = tempDir.resolve("hnsw_behavior.nvec");
    NvecWriter.write(original, file);
    HnswIndex restored = NvecReader.readHnsw(file);

    for (int q = 0; q < 10; q++) {
      float[] query = new float[dim];
      for (int d = 0; d < dim; d++) {
        query[d] = rng.nextFloat();
      }

      List<SearchResult> origRes = original.searchKnn(query, 10);
      List<SearchResult> restRes = restored.searchKnn(query, 10);

      assertThat(restRes).hasSameSizeAs(origRes);
      for (int k = 0; k < origRes.size(); k++) {
        SearchResult o = origRes.get(k);
        SearchResult r = restRes.get(k);
        assertThat(r.id()).isEqualTo(o.id());
        assertThat(r.distance())
            .isCloseTo(o.distance(), org.assertj.core.data.Offset.offset(1e-5f));
      }
    }
  }

  // ═════════════════════════════════════════════════════════════════════
  // TIER 3 — CORRUPTION RESISTANCE & STRUCTURAL INTEGRITY
  // ═════════════════════════════════════════════════════════════════════

  @Test
  @DisplayName("Single bit flip must be detected by CRC32C integrity gate")
  void testBitFlipDetectedByCrc() throws IOException {
    FlatIndex original = new FlatIndex(4, DistanceMetric.EUCLIDEAN);
    original.insert(1L, new float[] {1.0f, 2.0f, 3.0f, 4.0f});

    Path file = tempDir.resolve("bitflip.nvec");
    NvecWriter.write(original, file);

    byte[] bytes = Files.readAllBytes(file);
    // Flip a bit in the vector payload area (e.g. byte 40)
    bytes[40] ^= 0x01;
    Files.write(file, bytes);

    assertThatThrownBy(() -> NvecReader.read(file))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("CRC32C checksum mismatch");
  }

  @Test
  @DisplayName("Truncated file must be rejected by minimum size or CRC gate")
  void testTruncatedFile() throws IOException {
    FlatIndex original = new FlatIndex(4, DistanceMetric.EUCLIDEAN);
    original.insert(1L, new float[] {1.0f, 2.0f, 3.0f, 4.0f});

    Path file = tempDir.resolve("truncated.nvec");
    NvecWriter.write(original, file);

    byte[] bytes = Files.readAllBytes(file);
    // Truncate last 10 bytes
    byte[] truncated = new byte[bytes.length - 10];
    System.arraycopy(bytes, 0, truncated, 0, truncated.length);
    Files.write(file, truncated);

    assertThatThrownBy(() -> NvecReader.read(file)).isInstanceOf(CorruptIndexException.class);
  }

  @Test
  @DisplayName("File smaller than 40 bytes must be rejected immediately")
  void testFileSmallerThanMinimum() throws IOException {
    Path file = tempDir.resolve("tiny.nvec");
    Files.write(file, new byte[39]);

    assertThatThrownBy(() -> NvecReader.read(file))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("smaller than minimum valid NVEC file size");
  }

  @Test
  @DisplayName("Invalid magic bytes must be rejected")
  void testInvalidMagicBytes() throws IOException {
    FlatIndex original = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    Path file = tempDir.resolve("bad_magic.nvec");
    NvecWriter.write(original, file);

    byte[] bytes = Files.readAllBytes(file);
    bytes[0] = 'B';
    bytes[1] = 'A';
    bytes[2] = 'D';
    bytes[3] = '!';
    Files.write(file, bytes);

    assertThatThrownBy(() -> NvecReader.read(file))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Invalid NVEC magic bytes");
  }

  @Test
  @DisplayName("Unsupported version must throw UnsupportedVersionException")
  void testUnsupportedVersion() throws IOException {
    FlatIndex original = new FlatIndex(2, DistanceMetric.EUCLIDEAN);
    Path file = tempDir.resolve("bad_version.nvec");
    NvecWriter.write(original, file);

    byte[] bytes = Files.readAllBytes(file);
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putShort(4, (short) 99);
    Files.write(file, bytes);

    assertThatThrownBy(() -> NvecReader.read(file))
        .isInstanceOf(UnsupportedVersionException.class)
        .hasMessageContaining("Unsupported NVEC format version: 99");
  }

  @Test
  @DisplayName(
      "Valid CRC with corrupt huge dimension must be rejected BEFORE allocation (pre-allocation bounds check)")
  void testValidCrcCorruptDimensionPreAllocation() throws IOException {
    // Craft a validly check-summed 40-byte file that claims dimension = Integer.MAX_VALUE
    ByteBuffer buffer = ByteBuffer.allocate(40).order(ByteOrder.LITTLE_ENDIAN);
    buffer.put(new byte[] {'N', 'V', 'E', 'C'});
    buffer.putShort((short) 1);
    buffer.put((byte) 1); // endianness
    buffer.put((byte) 1); // FLAT
    buffer.put((byte) 1); // EUCLIDEAN
    buffer.put(new byte[] {0, 0, 0}); // reserved
    buffer.putInt(Integer.MAX_VALUE); // CRAZY DIMENSION!
    buffer.putInt(100); // 100 vectors
    buffer.put(new byte[12]); // padding
    buffer.putInt(0); // metadata_length = 0

    // Compute valid CRC over the first 36 bytes
    CRC32C crc = new CRC32C();
    crc.update(buffer.array(), 0, 36);
    buffer.putInt(36, (int) crc.getValue());

    Path file = tempDir.resolve("huge_dim_valid_crc.nvec");
    Files.write(file, buffer.array());

    // Reader must reject BEFORE attempting float[Integer.MAX_VALUE * 100] allocation!
    assertThatThrownBy(() -> NvecReader.read(file))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("overflow");
  }

  @Test
  @DisplayName(
      "Valid CRC with file size smaller than structural minimum must be rejected pre-allocation")
  void testValidCrcFileSizeSmallerThanExpected() throws IOException {
    // 40 bytes file with valid CRC claiming 1000 vectors of dim 128 (needs > 500 KB)
    ByteBuffer buffer = ByteBuffer.allocate(40).order(ByteOrder.LITTLE_ENDIAN);
    buffer.put(new byte[] {'N', 'V', 'E', 'C'});
    buffer.putShort((short) 1);
    buffer.put((byte) 1);
    buffer.put((byte) 1); // FLAT
    buffer.put((byte) 1); // EUCLIDEAN
    buffer.put(new byte[] {0, 0, 0});
    buffer.putInt(128); // dim 128
    buffer.putInt(1000); // 1000 vectors
    buffer.put(new byte[12]);
    buffer.putInt(0); // metaLen = 0

    CRC32C crc = new CRC32C();
    crc.update(buffer.array(), 0, 36);
    buffer.putInt(36, (int) crc.getValue());

    Path file = tempDir.resolve("too_small_valid_crc.nvec");
    Files.write(file, buffer.array());

    assertThatThrownBy(() -> NvecReader.read(file))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("FLAT index file size smaller than structural minimum");
  }

  @Test
  @DisplayName("Valid CRC with invalid HNSW topology (out of bounds neighbor ID) must be rejected")
  void testValidCrcInvalidTopologyNeighborId() throws IOException {
    HnswIndex original = new HnswIndex(2, DistanceMetric.EUCLIDEAN, HnswConfig.defaultConfig());
    original.insert(1L, new float[] {0.1f, 0.2f});
    original.insert(2L, new float[] {0.3f, 0.4f});

    Path file = tempDir.resolve("bad_nbr.nvec");
    NvecWriter.write(original, file);

    byte[] bytes = Files.readAllBytes(file);
    // Find the topology block and corrupt neighbor ID to 9999 (out of bounds for size=2)
    // Header (32) + Meta (28) + Vectors (2*2*4 = 16) + IDs (2*8 = 16) = 92
    // Topology begins at offset 92: node 0: level (1B), degree (2B), neighbor 0 (4B at offset 95)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(95, 9999);

    // Recompute CRC so CRC is 100% VALID, but topology is corrupted
    CRC32C crc = new CRC32C();
    crc.update(bytes, 0, bytes.length - 4);
    ByteBuffer.wrap(bytes)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putInt(bytes.length - 4, (int) crc.getValue());
    Files.write(file, bytes);

    assertThatThrownBy(() -> NvecReader.read(file))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Neighbor ID out of bounds: 9999");
  }

  @Test
  @DisplayName("Valid CRC with HNSW topology self-loop must be rejected")
  void testValidCrcInvalidTopologySelfLoop() throws IOException {
    HnswIndex original = new HnswIndex(2, DistanceMetric.EUCLIDEAN, HnswConfig.defaultConfig());
    original.insert(1L, new float[] {0.1f, 0.2f});
    original.insert(2L, new float[] {0.3f, 0.4f});

    Path file = tempDir.resolve("self_loop.nvec");
    NvecWriter.write(original, file);

    byte[] bytes = Files.readAllBytes(file);
    // Node 0 neighbor corrupted to 0 (self-loop!)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(95, 0);

    CRC32C crc = new CRC32C();
    crc.update(bytes, 0, bytes.length - 4);
    ByteBuffer.wrap(bytes)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putInt(bytes.length - 4, (int) crc.getValue());
    Files.write(file, bytes);

    assertThatThrownBy(() -> NvecReader.read(file))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Self-loop detected");
  }

  @Test
  @DisplayName("Forward compatibility: safely skips unknown metadata extensions")
  void testForwardCompatibleMetadataExtension() throws IOException {
    int dim = 2;
    FlatIndex original = new FlatIndex(dim, DistanceMetric.EUCLIDEAN);
    original.insert(1L, new float[] {1.0f, 2.0f});

    // Manually craft file with metadata_length = 8 (8 bytes of unknown future extension)
    // 32 header + 4 metaLen + 8 extension + 8 vectors + 8 IDs + 4 CRC = 64 bytes
    ByteBuffer buffer = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    buffer.put(new byte[] {'N', 'V', 'E', 'C'});
    buffer.putShort((short) 1);
    buffer.put((byte) 1); // endianness
    buffer.put((byte) 1); // FLAT
    buffer.put((byte) 1); // EUCLIDEAN
    buffer.put(new byte[] {0, 0, 0});
    buffer.putInt(dim);
    buffer.putInt(1); // 1 vector
    buffer.put(new byte[12]); // padding

    buffer.putInt(8); // metadata_length = 8 bytes of extension!
    buffer.putLong(0xDEADBEEFCAFEBABEL); // unknown extension payload

    buffer.putFloat(1.0f);
    buffer.putFloat(2.0f);
    buffer.putLong(1L);

    CRC32C crc = new CRC32C();
    crc.update(buffer.array(), 0, 60);
    buffer.putInt((int) crc.getValue());

    Path file = tempDir.resolve("forward_meta.nvec");
    Files.write(file, buffer.array());

    // Reader should skip the 8 extension bytes and successfully restore the index
    FlatIndex restored = NvecReader.readFlat(file);
    assertThat(restored.size()).isEqualTo(1);
    assertThat(restored.getVector(0)).containsExactly(1.0f, 2.0f);
  }
}
