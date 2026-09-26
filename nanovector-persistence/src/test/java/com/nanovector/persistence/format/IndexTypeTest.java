package com.nanovector.persistence.format;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class IndexTypeTest {

  @Test
  @DisplayName("Should return valid codes for FLAT and HNSW")
  void testCodes() {
    assertThat(IndexType.FLAT.code()).isEqualTo((byte) 1);
    assertThat(IndexType.HNSW.code()).isEqualTo((byte) 2);
  }

  @Test
  @DisplayName("Should resolve IndexType from valid byte codes")
  void testFromCode() {
    assertThat(IndexType.fromCode((byte) 1)).isEqualTo(IndexType.FLAT);
    assertThat(IndexType.fromCode((byte) 2)).isEqualTo(IndexType.HNSW);
  }

  @Test
  @DisplayName("Should throw IllegalArgumentException on unknown code")
  void testUnknownCode() {
    assertThatThrownBy(() -> IndexType.fromCode((byte) 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Unknown index type code: 0");

    assertThatThrownBy(() -> IndexType.fromCode((byte) 3))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Unknown index type code: 3");

    assertThatThrownBy(() -> IndexType.fromCode((byte) -1))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
