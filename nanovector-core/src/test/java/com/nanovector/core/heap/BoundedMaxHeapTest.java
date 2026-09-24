package com.nanovector.core.heap;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.model.SearchResult;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class BoundedMaxHeapTest {

  @Test
  @DisplayName("Heap: Retains top K smallest distances from 1,000 random items")
  void testTopKSmallestMatchesArraysSort() {
    int totalItems = 1000;
    int k = 10;
    Random rng = new Random(42);

    List<SearchResult> allItems = new ArrayList<>(totalItems);
    BoundedMaxHeap heap = new BoundedMaxHeap(k);

    for (int i = 0; i < totalItems; i++) {
      float distance = rng.nextFloat() * 100.0f;
      long id = i + 1;
      allItems.add(new SearchResult(id, distance));
      heap.offer(id, distance);
    }

    // Independent sorting of all candidates
    Collections.sort(allItems);
    List<SearchResult> expectedTopK = allItems.subList(0, k);

    List<SearchResult> actualTopK = heap.toSortedList();

    assertThat(actualTopK).hasSize(k);
    for (int i = 0; i < k; i++) {
      assertThat(actualTopK.get(i).id()).isEqualTo(expectedTopK.get(i).id());
      assertThat(actualTopK.get(i).distance()).isEqualTo(expectedTopK.get(i).distance());
    }
  }

  @Test
  @DisplayName("Heap: Deterministic tie-breaking on identical distances orders by ID ascending")
  void testDeterministicTieBreaking() {
    BoundedMaxHeap heap = new BoundedMaxHeap(3);

    // All have identical distance 1.5f, but different IDs
    heap.offer(300L, 1.5f);
    heap.offer(100L, 1.5f);
    heap.offer(200L, 1.5f);
    heap.offer(400L, 1.5f); // Should be discarded because 400 > 300

    List<SearchResult> results = heap.toSortedList();

    assertThat(results).hasSize(3);
    assertThat(results.get(0).id()).isEqualTo(100L);
    assertThat(results.get(1).id()).isEqualTo(200L);
    assertThat(results.get(2).id()).isEqualTo(300L);
  }

  @Test
  @DisplayName("Heap: Capacity validation and empty state")
  void testCapacityValidation() {
    assertThatThrownBy(() -> new BoundedMaxHeap(0)).isInstanceOf(IllegalArgumentException.class);

    BoundedMaxHeap emptyHeap = new BoundedMaxHeap(5);
    assertThat(emptyHeap.toSortedList()).isEmpty();
  }
}
