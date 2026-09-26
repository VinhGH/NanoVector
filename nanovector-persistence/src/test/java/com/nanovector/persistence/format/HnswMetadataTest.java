package com.nanovector.persistence.format;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.persistence.exception.CorruptIndexException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class HnswMetadataTest {

  @Test
  @DisplayName("Should successfully construct and round-trip valid HnswMetadata")
  void testValidRoundTrip() throws CorruptIndexException {
    HnswMetadata meta = new HnswMetadata(16, 32, 200, 50, 4, 42);

    ByteBuffer buffer = ByteBuffer.allocate(24);
    meta.write(buffer);
    buffer.flip();

    HnswMetadata restored = HnswMetadata.read(buffer);
    assertThat(restored).isEqualTo(meta);
    assertThat(restored.m()).isEqualTo(16);
    assertThat(restored.m0()).isEqualTo(32);
    assertThat(restored.efConstruction()).isEqualTo(200);
    assertThat(restored.defaultEfSearch()).isEqualTo(50);
    assertThat(restored.maxLevel()).isEqualTo(4);
    assertThat(restored.entryPointId()).isEqualTo(42);
  }

  @Test
  @DisplayName("Should support empty index metadata (-1 maxLevel and entryPointId)")
  void testEmptyIndexMetadata() throws CorruptIndexException {
    HnswMetadata meta = new HnswMetadata(16, 32, 100, 40, -1, -1);

    ByteBuffer buffer = ByteBuffer.allocate(24);
    meta.write(buffer);
    buffer.flip();

    HnswMetadata restored = HnswMetadata.read(buffer);
    assertThat(restored).isEqualTo(meta);
    assertThat(restored.maxLevel()).isEqualTo(-1);
    assertThat(restored.entryPointId()).isEqualTo(-1);
  }

  @Test
  @DisplayName("Should preserve caller ByteOrder when writing and reading")
  void testPreservesByteOrder() throws CorruptIndexException {
    HnswMetadata meta = new HnswMetadata(16, 32, 200, 50, 0, 0);

    ByteBuffer buffer = ByteBuffer.allocate(32).order(ByteOrder.BIG_ENDIAN);
    meta.write(buffer);
    assertThat(buffer.order()).isEqualTo(ByteOrder.BIG_ENDIAN);

    buffer.flip();
    HnswMetadata restored = HnswMetadata.read(buffer);
    assertThat(buffer.order()).isEqualTo(ByteOrder.BIG_ENDIAN);
    assertThat(restored).isEqualTo(meta);
  }

  @Test
  @DisplayName("Should throw IllegalArgumentException on invalid parameter invariants")
  void testInvalidInvariants() {
    // m < 2
    assertThatThrownBy(() -> new HnswMetadata(1, 32, 200, 50, 0, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("m must be at least 2");

    // m0 < m
    assertThatThrownBy(() -> new HnswMetadata(16, 15, 200, 50, 0, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("m0 must be at least m");

    // efConstruction <= 0
    assertThatThrownBy(() -> new HnswMetadata(16, 32, 0, 50, 0, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("efConstruction must be positive");

    // defaultEfSearch <= 0
    assertThatThrownBy(() -> new HnswMetadata(16, 32, 200, 0, 0, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultEfSearch must be positive");

    // maxLevel < -1
    assertThatThrownBy(() -> new HnswMetadata(16, 32, 200, 50, -2, -1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("maxLevel must be >= -1");

    // entryPointId < -1
    assertThatThrownBy(() -> new HnswMetadata(16, 32, 200, 50, -1, -2))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("entryPointId must be >= -1");

    // Inconsistent empty values (one is -1, other is not)
    assertThatThrownBy(() -> new HnswMetadata(16, 32, 200, 50, -1, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("maxLevel and entryPointId must both be -1");

    assertThatThrownBy(() -> new HnswMetadata(16, 32, 200, 50, 0, -1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("maxLevel and entryPointId must both be -1");
  }

  @Test
  @DisplayName("Should throw CorruptIndexException on truncated buffer or corrupted values")
  void testCorruptedBufferRead() {
    ByteBuffer truncated = ByteBuffer.allocate(20);
    assertThatThrownBy(() -> HnswMetadata.read(truncated))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Insufficient buffer size");

    ByteBuffer corruptValues = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    corruptValues.putInt(1); // invalid m (< 2)
    corruptValues.putInt(32);
    corruptValues.putInt(200);
    corruptValues.putInt(50);
    corruptValues.putInt(0);
    corruptValues.putInt(0);
    corruptValues.flip();

    assertThatThrownBy(() -> HnswMetadata.read(corruptValues))
        .isInstanceOf(CorruptIndexException.class)
        .hasMessageContaining("Invalid HNSW metadata values");
  }
}
