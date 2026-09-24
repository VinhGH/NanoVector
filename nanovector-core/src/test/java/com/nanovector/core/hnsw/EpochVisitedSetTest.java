package com.nanovector.core.hnsw;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class EpochVisitedSetTest {

  @Test
  @DisplayName("EpochVisitedSet: Marks and identifies visited nodes correctly")
  void testBasicVisitAndQuery() {
    EpochVisitedSet set = new EpochVisitedSet(16);

    assertThat(set.isVisited(5)).isFalse();
    assertThat(set.isVisited(10)).isFalse();

    set.markVisited(5);

    assertThat(set.isVisited(5)).isTrue();
    assertThat(set.isVisited(4)).isFalse();
    assertThat(set.isVisited(6)).isFalse();
    assertThat(set.isVisited(10)).isFalse();
  }

  @Test
  @DisplayName("EpochVisitedSet: nextEpoch resets visited state in O(1) without modifying array")
  void testNextEpochResetsState() {
    EpochVisitedSet set = new EpochVisitedSet(16);
    set.markVisited(2);
    set.markVisited(7);
    assertThat(set.isVisited(2)).isTrue();
    assertThat(set.isVisited(7)).isTrue();

    // Advance epoch
    set.nextEpoch();

    // Previous marks must now be considered unvisited
    assertThat(set.isVisited(2)).isFalse();
    assertThat(set.isVisited(7)).isFalse();

    // Mark in the new epoch
    set.markVisited(2);
    assertThat(set.isVisited(2)).isTrue();
    assertThat(set.isVisited(7)).isFalse();
  }

  @Test
  @DisplayName(
      "EpochVisitedSet: Dynamic capacity expansion preserves state and accommodates large IDs")
  void testDynamicResizing() {
    EpochVisitedSet set = new EpochVisitedSet(4);
    set.markVisited(2);

    // Mark a much larger ID that triggers resizing
    set.markVisited(500);

    assertThat(set.capacity()).isGreaterThanOrEqualTo(501);
    assertThat(set.isVisited(2)).isTrue();
    assertThat(set.isVisited(500)).isTrue();
    assertThat(set.isVisited(300)).isFalse();
  }

  @Test
  @DisplayName("EpochVisitedSet: Handles Integer.MAX_VALUE overflow safely by resetting")
  void testEpochOverflowHandling() {
    EpochVisitedSet set = new EpochVisitedSet(16);
    set.setCurrentEpochForTesting(Integer.MAX_VALUE);
    set.markVisited(3);
    assertThat(set.isVisited(3)).isTrue();

    // Calling nextEpoch at MAX_VALUE should reset epoch to 1 and clear the array
    set.nextEpoch();

    assertThat(set.currentEpoch()).isEqualTo(1);
    assertThat(set.isVisited(3)).isFalse();

    // Should function normally in epoch 1
    set.markVisited(3);
    assertThat(set.isVisited(3)).isTrue();
  }

  @Test
  @DisplayName("EpochVisitedSet: Negative indices throw IndexOutOfBoundsException")
  void testNegativeIndices() {
    EpochVisitedSet set = new EpochVisitedSet(16);

    assertThatThrownBy(() -> set.isVisited(-1)).isInstanceOf(IndexOutOfBoundsException.class);

    assertThatThrownBy(() -> set.markVisited(-1)).isInstanceOf(IndexOutOfBoundsException.class);
  }
}
