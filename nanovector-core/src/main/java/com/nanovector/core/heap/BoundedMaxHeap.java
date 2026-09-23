package com.nanovector.core.heap;

import com.nanovector.core.model.SearchResult;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Fixed-capacity primitive Max-Heap for tracking Top-K nearest neighbors.
 * <p>
 * Under the distance minimization contract, the "worst" candidate in the Top-K
 * (largest distance, or largest ID upon distance tie) resides at the root (index 0).
 * <p>
 * Internal representations use primitive arrays ({@code long[]}, {@code float[]})
 * to completely eliminate allocations during candidate evaluations.
 */
public final class BoundedMaxHeap {

    private final int capacity;
    private final long[] ids;
    private final float[] distances;
    private int size;

    public BoundedMaxHeap(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("Capacity must be positive: " + capacity);
        }
        this.capacity = capacity;
        this.ids = new long[capacity];
        this.distances = new float[capacity];
        this.size = 0;
    }

    /**
     * Considers a candidate for inclusion in the Top-K heap.
     *
     * @param id       the external ID of the candidate
     * @param distance the calculated distance of the candidate
     * @return {@code true} if the candidate was accepted into the heap, {@code false} if discarded
     */
    public boolean offer(long id, float distance) {
        if (size < capacity) {
            ids[size] = id;
            distances[size] = distance;
            siftUp(size);
            size++;
            return true;
        }

        // When heap is full, compare candidate with current worst element at root
        if (isWorse(distances[0], ids[0], distance, id)) {
            // Root is worse than candidate -> replace root and sift down
            ids[0] = id;
            distances[0] = distance;
            siftDown(0);
            return true;
        }

        // Discard candidate
        return false;
    }

    /**
     * Checks if candidate (d1, id1) is "worse" than (d2, id2).
     * <p>
     * "Worse" means:
     * 1. Larger distance.
     * 2. If distances are identical, larger ID.
     */
    private boolean isWorse(float d1, long id1, float d2, long id2) {
        int cmp = Float.compare(d1, d2);
        if (cmp != 0) {
            return cmp > 0;
        }
        return id1 > id2;
    }

    private void siftUp(int index) {
        int current = index;
        long targetId = ids[current];
        float targetDistance = distances[current];

        while (current > 0) {
            int parent = (current - 1) >>> 1;
            if (isWorse(targetDistance, targetId, distances[parent], ids[parent])) {
                // Target is worse than parent -> bubble up to maintain Max-Heap
                ids[current] = ids[parent];
                distances[current] = distances[parent];
                current = parent;
            } else {
                break;
            }
        }
        ids[current] = targetId;
        distances[current] = targetDistance;
    }

    private void siftDown(int index) {
        int current = index;
        long targetId = ids[current];
        float targetDistance = distances[current];
        int half = size >>> 1;

        while (current < half) {
            int child = (current << 1) + 1; // Left child
            int right = child + 1;

            if (right < size && isWorse(distances[right], ids[right], distances[child], ids[child])) {
                child = right;
            }

            if (isWorse(distances[child], ids[child], targetDistance, targetId)) {
                ids[current] = ids[child];
                distances[current] = distances[child];
                current = child;
            } else {
                break;
            }
        }
        ids[current] = targetId;
        distances[current] = targetDistance;
    }

    /**
     * Extracts all elements into a list sorted in ascending order (closest/best first).
     *
     * @return sorted list of {@link SearchResult}
     */
    public List<SearchResult> toSortedList() {
        if (size == 0) {
            return Collections.emptyList();
        }

        SearchResult[] temp = new SearchResult[size];
        int count = size;

        // Extract max into temp from back to front, achieving ascending order
        for (int i = count - 1; i >= 0; i--) {
            temp[i] = new SearchResult(ids[0], distances[0]);

            // Move last element to root and sift down
            size--;
            if (size > 0) {
                ids[0] = ids[size];
                distances[0] = distances[size];
                siftDown(0);
            }
        }

        List<SearchResult> result = new ArrayList<>(count);
        Collections.addAll(result, temp);
        return result;
    }

    public int size() {
        return size;
    }

    public int capacity() {
        return capacity;
    }

    public float peekMaxDistance() {
        if (size == 0) {
            throw new IllegalStateException("Heap is empty");
        }
        return distances[0];
    }
}
