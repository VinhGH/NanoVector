package com.nanovector.core.storage;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.NoSuchElementException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class VectorStorageTest {

    @Test
    @DisplayName("Storage: Insert and retrieve roundtrip")
    void testInsertAndRetrieve() {
        VectorStorage storage = new VectorStorage(3);
        float[] v1 = {1.0f, 2.0f, 3.0f};
        float[] v2 = {4.0f, 5.0f, 6.0f};

        int idx1 = storage.insert(1001L, v1);
        int idx2 = storage.insert(1002L, v2);

        assertThat(idx1).isEqualTo(0);
        assertThat(idx2).isEqualTo(1);
        assertThat(storage.size()).isEqualTo(2);

        assertThat(storage.getExternalId(0)).isEqualTo(1001L);
        assertThat(storage.getExternalId(1)).isEqualTo(1002L);

        assertThat(storage.getInternalId(1001L)).isEqualTo(0);
        assertThat(storage.getInternalId(1002L)).isEqualTo(1);

        assertThat(storage.getVector(0)).containsExactly(v1);
        assertThat(storage.getVector(1)).containsExactly(v2);
    }

    @Test
    @DisplayName("Storage: Duplicate external ID must throw IllegalArgumentException")
    void testDuplicateExternalIdRejected() {
        VectorStorage storage = new VectorStorage(2);
        storage.insert(999L, new float[]{1.0f, 2.0f});

        assertThatThrownBy(() -> storage.insert(999L, new float[]{3.0f, 4.0f}))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Duplicate external ID: 999");
    }

    @Test
    @DisplayName("Storage: Dynamic capacity expansion works without data loss")
    void testDynamicResizing() {
        // Start with small capacity = 2
        VectorStorage storage = new VectorStorage(2, 2);

        for (int i = 0; i < 50; i++) {
            storage.insert(i, new float[]{(float) i, (float) (i * 2)});
        }

        assertThat(storage.size()).isEqualTo(50);
        assertThat(storage.capacity()).isGreaterThanOrEqualTo(50);

        for (int i = 0; i < 50; i++) {
            float[] retrieved = storage.getVector(i);
            assertThat(retrieved).containsExactly((float) i, (float) (i * 2));
            assertThat(storage.getExternalId(i)).isEqualTo(i);
        }
    }

    @Test
    @DisplayName("Storage: copyVector copies into destination array without allocation")
    void testCopyVector() {
        VectorStorage storage = new VectorStorage(3);
        storage.insert(10L, new float[]{7.0f, 8.0f, 9.0f});

        float[] dest = new float[3];
        storage.copyVector(0, dest);

        assertThat(dest).containsExactly(7.0f, 8.0f, 9.0f);
    }

    @Test
    @DisplayName("Storage: Non-existent external ID throws NoSuchElementException")
    void testNonExistentExternalId() {
        VectorStorage storage = new VectorStorage(2);
        assertThatThrownBy(() -> storage.getInternalId(12345L))
                .isInstanceOf(NoSuchElementException.class);
    }
}
