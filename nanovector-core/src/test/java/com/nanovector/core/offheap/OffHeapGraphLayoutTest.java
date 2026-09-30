package com.nanovector.core.offheap;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.hnsw.HnswConfig;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class OffHeapGraphLayoutTest {

  @Test
  @DisplayName("Should initialize with correct stride and non-zero native memory")
  void shouldInitializeWithCorrectStride() {
    HnswConfig config = new HnswConfig(16, 32, 100, 50, 1.0 / Math.log(16), null);
    try (OffHeapGraphLayout layout = new OffHeapGraphLayout(config, 64)) {
      assertThat(layout.nodeCount()).isZero();
      assertThat(layout.capacity()).isGreaterThanOrEqualTo(64);
      assertThat(layout.l0MaxSlots()).isEqualTo(config.m0() + 8);
      assertThat(layout.upperMaxSlots()).isEqualTo(config.m() + 8);
      assertThat(layout.nativeAllocatedBytes()).isPositive();
    }
  }

  @Test
  @DisplayName("Should add nodes with various maxLevels and read/write Layer 0 neighbors")
  void shouldAddNodesAndManipulateLayer0Neighbors() {
    HnswConfig config = new HnswConfig(8, 16, 50, 50, 1.0 / Math.log(8), null);
    try (OffHeapGraphLayout layout = new OffHeapGraphLayout(config, 10)) {
      layout.addNode(0, 0);
      layout.addNode(1, 2);
      layout.addNode(2, 0);

      assertThat(layout.nodeCount()).isEqualTo(3);
      assertThat(layout.maxLevel(0)).isEqualTo(0);
      assertThat(layout.maxLevel(1)).isEqualTo(2);
      assertThat(layout.maxLevel(2)).isEqualTo(0);

      // Node 0: add neighbors
      layout.addNeighbor(0, 0, 1);
      layout.addNeighbor(0, 0, 2);
      assertThat(layout.degree(0, 0)).isEqualTo(2);
      assertThat(layout.getNeighbor(0, 0, 0)).isEqualTo(1);
      assertThat(layout.getNeighbor(0, 0, 1)).isEqualTo(2);
      assertThat(layout.hasNeighbor(0, 0, 1)).isTrue();
      assertThat(layout.hasNeighbor(0, 0, 2)).isTrue();
      assertThat(layout.hasNeighbor(0, 0, 3)).isFalse();
      assertThat(layout.getNeighbors(0, 0)).containsExactly(1, 2);

      // Remove neighbor
      layout.removeNeighbor(0, 0, 1);
      assertThat(layout.degree(0, 0)).isEqualTo(1);
      assertThat(layout.getNeighbors(0, 0)).containsExactly(2);

      // Set neighbors
      layout.setNeighbors(0, 0, new int[] {1, 2});
      assertThat(layout.getNeighbors(0, 0)).containsExactly(1, 2);
    }
  }

  @Test
  @DisplayName("Should support upper layers access and manipulation")
  void shouldSupportUpperLayers() {
    HnswConfig config = new HnswConfig(8, 16, 50, 50, 1.0 / Math.log(8), null);
    try (OffHeapGraphLayout layout = new OffHeapGraphLayout(config, 10)) {
      layout.addNode(0, 2);
      layout.addNode(1, 1);

      // Layer 1
      layout.addNeighbor(0, 1, 1);
      assertThat(layout.degree(0, 1)).isEqualTo(1);
      assertThat(layout.getNeighbors(0, 1)).containsExactly(1);

      // Layer 2
      layout.addNeighbor(0, 2, 99);
      assertThat(layout.degree(0, 2)).isEqualTo(1);
      assertThat(layout.getNeighbors(0, 2)).containsExactly(99);

      // Out of bounds layer check
      assertThatThrownBy(() -> layout.degree(1, 2))
          .isInstanceOf(IndexOutOfBoundsException.class)
          .hasMessageContaining("out of bounds for node 1");
    }
  }

  @Test
  @DisplayName("Should dynamically expand capacity when adding more nodes than initialCapacity")
  void shouldExpandCapacityDynamically() {
    HnswConfig config = new HnswConfig(4, 8, 20, 20, 1.0 / Math.log(4), null);
    try (OffHeapGraphLayout layout = new OffHeapGraphLayout(config, 2)) {
      for (int i = 0; i < 20; i++) {
        layout.addNode(i, i % 3);
        layout.addNeighbor(i, 0, (i + 1) % 20);
      }
      assertThat(layout.nodeCount()).isEqualTo(20);
      assertThat(layout.capacity()).isGreaterThanOrEqualTo(20);

      for (int i = 0; i < 20; i++) {
        assertThat(layout.degree(i, 0)).isEqualTo(1);
        assertThat(layout.getNeighbor(i, 0, 0)).isEqualTo((i + 1) % 20);
      }
    }
  }

  @Test
  @DisplayName("Should enforce bounds checks and throw descriptive exceptions")
  void shouldEnforceBoundsChecks() {
    HnswConfig config = new HnswConfig(8, 16, 50, 50, 1.0 / Math.log(8), null);
    try (OffHeapGraphLayout layout = new OffHeapGraphLayout(config, 4)) {
      layout.addNode(0, 1);

      // Non-sequential add
      assertThatThrownBy(() -> layout.addNode(2, 0))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Node internalId must be 1");

      // Negative maxLevel
      assertThatThrownBy(() -> layout.addNode(1, -1))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("non-negative");

      // Node out of bounds
      assertThatThrownBy(() -> layout.degree(5, 0)).isInstanceOf(IndexOutOfBoundsException.class);

      // Neighbor index out of bounds
      assertThatThrownBy(() -> layout.getNeighbor(0, 0, 0))
          .isInstanceOf(IndexOutOfBoundsException.class);
    }
  }

  @Test
  @DisplayName("Should operate cleanly with externally provided Arena and respect close()")
  void shouldOperateWithExternalArena() {
    HnswConfig config = new HnswConfig(4, 8, 20, 20, 1.0 / Math.log(4), null);
    try (Arena arena = Arena.ofConfined()) {
      OffHeapGraphLayout layout = new OffHeapGraphLayout(config, 8, arena);
      layout.addNode(0, 0);
      layout.addNeighbor(0, 0, 42);
      assertThat(layout.getNeighbors(0, 0)).containsExactly(42);

      layout.close();
      assertThatThrownBy(() -> layout.addNode(1, 0))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("closed");
    }
  }
}
