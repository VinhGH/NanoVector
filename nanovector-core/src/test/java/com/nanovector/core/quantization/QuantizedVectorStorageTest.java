package com.nanovector.core.quantization;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.NoSuchElementException;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class QuantizedVectorStorageTest {

  @Test
  @DisplayName("Basic insertion and capacity expansion")
  void testBasicInsertionAndExpansion() {
    int dim = 4;
    QuantizedVectorStorage storage = new QuantizedVectorStorage(dim, 2);

    assertThat(storage.size()).isEqualTo(0);
    assertThat(storage.dimension()).isEqualTo(dim);
    assertThat(storage.capacity()).isEqualTo(2);

    int id0 = storage.insert(100L, new float[] {1.0f, 2.0f, 3.0f, 4.0f});
    int id1 = storage.insert(200L, new float[] {10.0f, 20.0f, 30.0f, 40.0f});

    assertThat(id0).isEqualTo(0);
    assertThat(id1).isEqualTo(1);
    assertThat(storage.size()).isEqualTo(2);
    assertThat(storage.capacity()).isEqualTo(2);

    // Triggers capacity expansion
    int id2 = storage.insert(300L, new float[] {-1.0f, 0.0f, 1.0f, 2.0f});
    assertThat(id2).isEqualTo(2);
    assertThat(storage.size()).isEqualTo(3);
    assertThat(storage.capacity()).isGreaterThanOrEqualTo(4);

    assertThat(storage.contains(100L)).isTrue();
    assertThat(storage.contains(200L)).isTrue();
    assertThat(storage.contains(300L)).isTrue();
    assertThat(storage.contains(999L)).isFalse();

    assertThat(storage.getInternalId(100L)).isEqualTo(0);
    assertThat(storage.getInternalId(200L)).isEqualTo(1);
    assertThat(storage.getInternalId(300L)).isEqualTo(2);

    assertThat(storage.getExternalId(0)).isEqualTo(100L);
    assertThat(storage.getExternalId(1)).isEqualTo(200L);
    assertThat(storage.getExternalId(2)).isEqualTo(300L);
  }

  @Test
  @DisplayName("Zero-range constant vector assigns scale 0 and minimum without NaN/Inf")
  void testZeroRangeVector() {
    int dim = 8;
    QuantizedVectorStorage storage = new QuantizedVectorStorage(dim);

    float[] constantVector = new float[] {0.75f, 0.75f, 0.75f, 0.75f, 0.75f, 0.75f, 0.75f, 0.75f};
    int id = storage.insert(42L, constantVector);

    assertThat(storage.getMin(id)).isEqualTo(0.75f);
    assertThat(storage.getScale(id)).isEqualTo(0.0f);
    assertThat(Float.isFinite(storage.getMin(id))).isTrue();
    assertThat(Float.isFinite(storage.getScale(id))).isTrue();

    byte[] buf = storage.vectorBuffer();
    int offset = storage.getOffset(id);
    for (int i = 0; i < dim; i++) {
      assertThat(buf[offset + i]).isEqualTo((byte) 0);
    }

    float[] reconstructed = storage.getVector(id);
    for (int i = 0; i < dim; i++) {
      assertThat(reconstructed[i]).isEqualTo(0.75f);
    }
  }

  @Test
  @DisplayName("Reconstructed vector satisfies theoretical SQ8 bound")
  void testDequantizationFidelity() {
    int dim = 5;
    QuantizedVectorStorage storage = new QuantizedVectorStorage(dim);

    float[] vector = new float[] {-10.0f, -2.5f, 0.0f, 5.5f, 15.0f};
    int id = storage.insert(1L, vector);

    float min = storage.getMin(id);
    float scale = storage.getScale(id);
    assertThat(min).isEqualTo(-10.0f);
    assertThat(scale).isEqualTo(25.0f / 255.0f);

    float[] reconstructed = storage.getVector(id);
    float maxTol = scale / 2.0f + 1e-6f;

    for (int i = 0; i < dim; i++) {
      assertThat(Math.abs(vector[i] - reconstructed[i])).isLessThanOrEqualTo(maxTol);
    }

    float[] copied = new float[dim];
    storage.copyVector(id, copied);
    assertThat(copied).containsExactly(reconstructed);

    QuantizedVector qVec = storage.getQuantizedVector(id);
    assertThat(qVec.min()).isEqualTo(min);
    assertThat(qVec.scale()).isEqualTo(scale);
    assertThat(qVec.dimension()).isEqualTo(dim);

    byte[] copiedBytes = new byte[dim];
    storage.copyQuantizedVector(id, copiedBytes);
    assertThat(copiedBytes).containsExactly(qVec.data());
  }

  @Test
  @DisplayName("Rejects duplicate external IDs, invalid dimensions, and non-finite vectors")
  void testValidationRejections() {
    int dim = 3;
    QuantizedVectorStorage storage = new QuantizedVectorStorage(dim);

    storage.insert(10L, new float[] {1.0f, 2.0f, 3.0f});

    assertThatThrownBy(() -> storage.insert(10L, new float[] {4.0f, 5.0f, 6.0f}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Duplicate external ID");

    assertThatThrownBy(() -> storage.insert(11L, new float[] {1.0f, 2.0f}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dimension");

    assertThatThrownBy(() -> storage.insert(12L, new float[] {1.0f, Float.NaN, 3.0f}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("non-finite");

    assertThatThrownBy(() -> storage.insert(13L, new float[] {1.0f, Float.POSITIVE_INFINITY, 3.0f}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("non-finite");
  }

  @Test
  @DisplayName("Throws on invalid internal IDs and missing external IDs")
  void testBoundsAndMissingKeys() {
    QuantizedVectorStorage storage = new QuantizedVectorStorage(4);
    storage.insert(100L, new float[] {1.0f, 2.0f, 3.0f, 4.0f});

    assertThatThrownBy(() -> storage.getInternalId(999L))
        .isInstanceOf(NoSuchElementException.class);

    assertThatThrownBy(() -> storage.getExternalId(-1))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> storage.getExternalId(1))
        .isInstanceOf(IndexOutOfBoundsException.class);

    assertThatThrownBy(() -> storage.getOffset(1)).isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> storage.getMin(1)).isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> storage.getScale(1)).isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> storage.getVector(1)).isInstanceOf(IndexOutOfBoundsException.class);
  }

  @Test
  @DisplayName("Pre-populated constructor restores state and prevents duplicate external IDs")
  void testPrePopulatedConstructor() {
    int dim = 2;
    int size = 2;
    byte[] vectors = new byte[] {0, 10, 20, 30};
    float[] mins = new float[] {0.0f, 1.0f};
    float[] scales = new float[] {0.1f, 0.2f};
    long[] externalIds = new long[] {500L, 600L};

    QuantizedVectorStorage storage =
        new QuantizedVectorStorage(dim, size, vectors, mins, scales, externalIds);

    assertThat(storage.size()).isEqualTo(2);
    assertThat(storage.dimension()).isEqualTo(2);
    assertThat(storage.getExternalId(0)).isEqualTo(500L);
    assertThat(storage.getExternalId(1)).isEqualTo(600L);
    assertThat(storage.getInternalId(500L)).isEqualTo(0);
    assertThat(storage.getInternalId(600L)).isEqualTo(1);

    long[] dups = new long[] {500L, 500L};
    assertThatThrownBy(() -> new QuantizedVectorStorage(dim, size, vectors, mins, scales, dups))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Duplicate external ID");
  }
}
