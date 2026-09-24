package com.nanovector.core.hnsw;

import java.util.Arrays;

/**
 * Epoch-based visited set tracking for graph search and routing operations.
 * <p>
 * Instead of allocating a new set or clearing a bitset for every query, this structure
 * increments an integer epoch counter ({@code currentEpoch++}). A node is considered visited
 * if its entry in {@code visitedEpoch[internalId] == currentEpoch}.
 * <p>
 * <b>Performance guarantee:</b> Guarantees zero heap allocation and zero clearing overhead
 * during normal search operations once capacity is sufficient to hold all internal node IDs.
 * <p>
 * Safely handles integer overflow of the epoch counter by resetting the array when
 * {@code currentEpoch} reaches {@link Integer#MAX_VALUE}.
 */
public final class EpochVisitedSet {

    private static final int DEFAULT_INITIAL_CAPACITY = 1024;

    private int[] visitedEpoch;
    private int currentEpoch;

    public EpochVisitedSet() {
        this(DEFAULT_INITIAL_CAPACITY);
    }

    public EpochVisitedSet(int initialCapacity) {
        if (initialCapacity <= 0) {
            throw new IllegalArgumentException("Initial capacity must be positive, but got: " + initialCapacity);
        }
        this.visitedEpoch = new int[initialCapacity];
        this.currentEpoch = 1;
    }

    /**
     * Checks if the given internal node ID has been visited in the current epoch.
     *
     * @param internalId internal index of the node
     * @return {@code true} if visited in this epoch, {@code false} otherwise
     */
    public boolean isVisited(int internalId) {
        if (internalId < 0) {
            throw new IndexOutOfBoundsException("Internal ID cannot be negative: " + internalId);
        }
        if (internalId >= visitedEpoch.length) {
            return false;
        }
        return visitedEpoch[internalId] == currentEpoch;
    }

    /**
     * Marks the given internal node ID as visited in the current epoch.
     * Automatically expands capacity if needed.
     *
     * @param internalId internal index of the node
     */
    public void markVisited(int internalId) {
        if (internalId < 0) {
            throw new IndexOutOfBoundsException("Internal ID cannot be negative: " + internalId);
        }
        ensureCapacity(internalId + 1);
        visitedEpoch[internalId] = currentEpoch;
    }

    /**
     * Advances to the next epoch in O(1) time without clearing the array.
     * <p>
     * If the epoch counter reaches {@link Integer#MAX_VALUE}, the array is reset
     * to prevent integer overflow.
     */
    public void nextEpoch() {
        if (currentEpoch == Integer.MAX_VALUE) {
            // Safe overflow reset: zero out array and reset epoch counter to 1
            Arrays.fill(visitedEpoch, 0);
            currentEpoch = 1;
        } else {
            currentEpoch++;
        }
    }

    /**
     * Ensures that the visited set can accommodate node IDs up to {@code minCapacity - 1}.
     *
     * @param minCapacity minimum required capacity
     */
    public void ensureCapacity(int minCapacity) {
        if (minCapacity > visitedEpoch.length) {
            int newCapacity = Math.max(minCapacity, visitedEpoch.length * 2);
            visitedEpoch = Arrays.copyOf(visitedEpoch, newCapacity);
        }
    }

    public int capacity() {
        return visitedEpoch.length;
    }

    public int currentEpoch() {
        return currentEpoch;
    }

    /**
     * Package-private method for testing overflow handling.
     */
    void setCurrentEpochForTesting(int epoch) {
        this.currentEpoch = epoch;
    }
}
