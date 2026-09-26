package com.nanovector.persistence.format;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.persistence.exception.CorruptIndexException;
import com.nanovector.persistence.exception.UnsupportedVersionException;
import java.io.IOException;
import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class NvecHeaderTest {

  @Test
  @DisplayName("Should successfully serialize and deserialize FLAT index header")
  void testRoundTripFlat() throws IOException {
    NvecHeader header = NvecHeader.of(IndexType.FLAT, DistanceMetric.EUCLIDEAN, 128, 1000);

    ByteBuffer buffer = ByteBuffer.allocate(32);
    header.write(buffer);
    buffer.flip();

    NvecHeader restored = NvecHeader.read(buffer);
    assertThat(restored).isEqualTo(header);
    assertThat(restored.version()).isEqualTo((short) 1);
    assertThat(restored.endianness()).isEqualTo((byte) 1);
    assertThat(restored.indexType()).isEqualTo(IndexType.FLAT);
    assertThat(restored.metric()).isEqualTo(DistanceMetric.EUCLIDEAN);
    assertThat(restored.dimension()).isEqualTo(128);
    assertThat(restored.vectorCount()).isEqualTo(1000);
  }

  @Test
  @DisplayName("Should successfully serialize and deserialize HNSW index header")
  void testRoundTripHnsw() throws IOException {
    NvecHeader header = NvecHeader.of(IndexType.HNSW, DistanceMetric.COSINE, 768, 50000);

    ByteBuffer buffer = ByteBuffer.allocate(32);
    header.write(buffer);
    buffer.flip();

    NvecHeader restored = NvecHeader.read(buffer);
    assertThat(restored).isEqualTo(header);
    assertThat(restored.version()).isEqualTo((short) 1);
    assertThat(restored.endianness()).isEqualTo((byte) 1);
    assertThat(restored.indexType()).isEqualTo(IndexType.HNSW);
    assertThat(restored.metric()).isEqualTo(DistanceMetric.COSINE);
    assertThat(restored.dimension()).isEqualTo(768);
    assertThat(restored.vectorCount()).isEqualTo(50000);
  }

  @Test
  @DisplayName("Should preserve caller ByteOrder when writing and reading")
  void testPreservesByteOrder() throws IOException {
    NvecHeader header = NvecHeader.of(IndexType.FLAT, DistanceMetric.DOT_PRODUCT, 64, 0);

    ByteBuffer buffer = ByteBuffer.allocate(64).order(ByteOrder.BIG_ENDIAN);
    header.write(buffer);
    assertThat(buffer.order()).isEqualTo(ByteOrder.BIG_ENDIAN);

    buffer.flip();
    NvecHeader restored = NvecHeader.read(buffer);
    assertThat(buffer.order()).isEqualTo(ByteOrder.BIG_ENDIAN);
    assertThat(restored).isEqualTo(header);
  }

  @Test
  @DisplayName("Should throw BufferOverflowException if write buffer has fewer than 32 bytes")
  void testWriteBufferOverflow() {
    NvecHeader header = NvecHeader.of(IndexType.FLAT, DistanceMetric.EUCLIDEAN, 128, 10);
    ByteBuffer buffer = ByteBuffer.allocate(31);
    assertThatThrownBy(() -> header.write(buffer)).isInstanceOf(BufferOverflowException.class);
  }

  @Test
  @DisplayName("Should throw CorruptIndexException if read buffer has fewer than 32 bytes")
  void testReadBufferTooSmall() {
    ByteBuffer buffer = ByteBuffer.allocate(31);
    assertThatThrownBy(() -> NvecHeader.read(buffer))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Insufficient buffer size");
  }

  @Test
  @DisplayName("Should reject invalid magic bytes with CorruptIndexException")
  void testCorruptedMagic() {
    ByteBuffer buffer = createValidHeaderBuffer();
    buffer.put(0, (byte) 'B');
    buffer.put(1, (byte) 'A');
    buffer.put(2, (byte) 'D');
    buffer.put(3, (byte) '!');

    assertThatThrownBy(() -> NvecHeader.read(buffer))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Invalid NVEC magic bytes");
  }

  @Test
  @DisplayName("Should reject unsupported version with UnsupportedVersionException")
  void testUnsupportedVersion() {
    ByteBuffer buffer = createValidHeaderBuffer();
    buffer.order(ByteOrder.LITTLE_ENDIAN);
    buffer.putShort(4, (short) 2);

    assertThatThrownBy(() -> NvecHeader.read(buffer))
        .isInstanceOf(UnsupportedVersionException.class)
        .hasMessageContaining("Unsupported NVEC format version: 2");
  }

  @Test
  @DisplayName("Should reject non-Little-Endian endianness marker")
  void testUnsupportedEndianness() {
    ByteBuffer buffer = createValidHeaderBuffer();
    buffer.put(6, (byte) 2); // Big-Endian marker

    assertThatThrownBy(() -> NvecHeader.read(buffer))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Unsupported endianness byte: 2");
  }

  @Test
  @DisplayName("Should reject invalid index type code")
  void testInvalidIndexType() {
    ByteBuffer buffer = createValidHeaderBuffer();
    buffer.put(7, (byte) 99);

    assertThatThrownBy(() -> NvecHeader.read(buffer))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Invalid index type code: 99");
  }

  @Test
  @DisplayName("Should reject invalid distance metric code")
  void testInvalidMetricCode() {
    ByteBuffer buffer = createValidHeaderBuffer();
    buffer.put(8, (byte) 99);

    assertThatThrownBy(() -> NvecHeader.read(buffer))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Invalid distance metric code: 99");
  }

  @Test
  @DisplayName("Should reject non-zero reserved bytes")
  void testNonZeroReservedBytes() {
    for (int offset = 9; offset <= 11; offset++) {
      ByteBuffer buffer = createValidHeaderBuffer();
      buffer.put(offset, (byte) 1);
      assertThatThrownBy(() -> NvecHeader.read(buffer))
          .isInstanceOf(CorruptIndexException.class)
          .hasMessageContaining("Non-zero reserved byte");
    }
  }

  @Test
  @DisplayName("Should reject non-positive dimension")
  void testNonPositiveDimension() {
    ByteBuffer buffer = createValidHeaderBuffer();
    buffer.order(ByteOrder.LITTLE_ENDIAN);
    buffer.putInt(12, 0);

    assertThatThrownBy(() -> NvecHeader.read(buffer))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Dimension must be positive");
  }

  @Test
  @DisplayName("Should reject negative vector count")
  void testNegativeVectorCount() {
    ByteBuffer buffer = createValidHeaderBuffer();
    buffer.order(ByteOrder.LITTLE_ENDIAN);
    buffer.putInt(16, -1);

    assertThatThrownBy(() -> NvecHeader.read(buffer))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Vector count must be non-negative");
  }

  @Test
  @DisplayName("Should reject non-zero header padding bytes")
  void testNonZeroPaddingBytes() {
    for (int offset = 20; offset < 32; offset++) {
      ByteBuffer buffer = createValidHeaderBuffer();
      buffer.put(offset, (byte) 0xFF);
      assertThatThrownBy(() -> NvecHeader.read(buffer))
          .isInstanceOf(CorruptIndexException.class)
          .hasMessageContaining("Non-zero header padding byte");
    }
  }

  @Test
  @DisplayName("Constructor should validate invariants")
  void testConstructorValidation() {
    assertThatThrownBy(
            () ->
                new NvecHeader(
                    (short) 2, (byte) 1, IndexType.FLAT, DistanceMetric.EUCLIDEAN, 128, 10))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Unsupported NVEC version");

    assertThatThrownBy(
            () ->
                new NvecHeader(
                    (short) 1, (byte) 2, IndexType.FLAT, DistanceMetric.EUCLIDEAN, 128, 10))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Unsupported endianness marker");

    assertThatThrownBy(
            () -> new NvecHeader((short) 1, (byte) 1, null, DistanceMetric.EUCLIDEAN, 128, 10))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("IndexType must not be null");

    assertThatThrownBy(() -> new NvecHeader((short) 1, (byte) 1, IndexType.FLAT, null, 128, 10))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("DistanceMetric must not be null");

    assertThatThrownBy(
            () ->
                new NvecHeader(
                    (short) 1, (byte) 1, IndexType.FLAT, DistanceMetric.EUCLIDEAN, 0, 10))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Dimension must be positive");

    assertThatThrownBy(
            () ->
                new NvecHeader(
                    (short) 1, (byte) 1, IndexType.FLAT, DistanceMetric.EUCLIDEAN, 128, -1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Vector count must be non-negative");
  }

  private ByteBuffer createValidHeaderBuffer() {
    NvecHeader header = NvecHeader.of(IndexType.FLAT, DistanceMetric.EUCLIDEAN, 128, 100);
    ByteBuffer buffer = ByteBuffer.allocate(32);
    header.write(buffer);
    buffer.flip();
    return buffer;
  }
}
