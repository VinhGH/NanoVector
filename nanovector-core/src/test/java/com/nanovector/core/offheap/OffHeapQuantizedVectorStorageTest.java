package com.nanovector.core.offheap;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.NoSuchElementException;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class OffHeapQuantizedVectorStorageTest {

  @Test
  @DisplayName("Should store vectors in off-heap segment and dequantize within expected precision")
  void shouldStoreAndDequantizeVectors() {
    int dim = 128;
    try (OffHeapQuantizedVectorStorage storage = new OffHeapQuantizedVectorStorage(dim, 10)) {
      float[] v1 = new float[dim];
      for (int i = 0; i < dim; i++) {
        v1[i] = (float) (i * 0.1);
      }

      int id1 = storage.add(1001L, v1);
      assertThat(id1).isEqualTo(0);
      assertThat(storage.size()).isEqualTo(1);
      assertThat(storage.contains(1001L)).isTrue();
      assertThat(storage.getInternalId(1001L)).isEqualTo(0);
      assertThat(storage.getExternalId(0)).isEqualTo(1001L);

      // Quantized byte buffer read
      byte[] quantized = storage.getQuantizedVector(0);
      assertThat(quantized).hasSize(dim);

      // Dequantize and verify numerical closeness
      float[] reconstructed = storage.getVector(0);
      assertThat(reconstructed).hasSize(dim);
      for (int i = 0; i < dim; i++) {
        assertThat(reconstructed[i]).isCloseTo(v1[i], org.assertj.core.data.Offset.offset(0.1f));
      }

      // Memory footprint check: 136 bytes/vector stride for D=128
      assertThat(storage.vectorStrideBytes()).isEqualTo(136L);
      assertThat(storage.nativeAllocatedBytes()).isGreaterThan(0L);
    }
  }

  @Test
  @DisplayName(
      "Should automatically expand off-heap memory capacity when inserting beyond initialCapacity")
  void shouldExpandCapacityDynamically() {
    int dim = 16;
    try (OffHeapQuantizedVectorStorage storage = new OffHeapQuantizedVectorStorage(dim, 2)) {
      for (int i = 0; i < 20; i++) {
        float[] v = new float[dim];
        v[0] = (float) i;
        storage.add(1000L + i, v);
      }

      assertThat(storage.size()).isEqualTo(20);
      assertThat(storage.capacity()).isGreaterThanOrEqualTo(20);

      for (int i = 0; i < 20; i++) {
        assertThat(storage.getExternalId(i)).isEqualTo(1000L + i);
        assertThat(storage.getVector(i)[0])
            .isCloseTo((float) i, org.assertj.core.data.Offset.offset(0.01f));
      }
    }
  }

  @Test
  @DisplayName("Should reject duplicate external IDs and invalid dimensions")
  void shouldRejectDuplicatesAndInvalidInput() {
    int dim = 8;
    try (OffHeapQuantizedVectorStorage storage = new OffHeapQuantizedVectorStorage(dim, 4)) {
      float[] v = new float[dim];
      storage.add(1L, v);

      assertThatThrownBy(() -> storage.add(1L, v))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("already exists");

      assertThatThrownBy(() -> storage.add(2L, new float[4]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("does not match storage dimension");

      assertThatThrownBy(() -> storage.getInternalId(999L))
          .isInstanceOf(NoSuchElementException.class);
    }
  }

  @Test
  @DisplayName("Should reject operations after close()")
  void shouldRejectAfterClose() {
    OffHeapQuantizedVectorStorage storage = new OffHeapQuantizedVectorStorage(4, 2);
    storage.add(1L, new float[] {1f, 2f, 3f, 4f});
    storage.close();

    assertThatThrownBy(() -> storage.add(2L, new float[] {1f, 2f, 3f, 4f}))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("closed");
  }
}
