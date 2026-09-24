package com.nanovector.core.hnsw;

import java.util.Arrays;

/**
 * Represents a single node in the HNSW multi-layer graph.
 *
 * <p>Each node stores its internal ID, the maximum layer it participates in, and an adjacency list
 * of neighbor internal IDs per layer ({@code int[][] neighbors}).
 *
 * <p><b>Design note:</b> {@code HnswNode} is a pure graph structure and has no dependency on {@link
 * com.nanovector.core.storage.VectorStorage} or distance calculations. Vector data is accessed
 * externally through {@code VectorStorage} by the orchestrating {@code HnswIndex}.
 */
public final class HnswNode {

  private final int internalId;
  private final int maxLevel;
  private final int[][] neighbors;

  /**
   * Creates a new HNSW node.
   *
   * @param internalId the internal index assigned by {@code VectorStorage}
   * @param maxLevel the highest layer this node participates in (0-based, inclusive)
   */
  public HnswNode(int internalId, int maxLevel) {
    if (internalId < 0) {
      throw new IllegalArgumentException("Internal ID must be non-negative: " + internalId);
    }
    if (maxLevel < 0) {
      throw new IllegalArgumentException("Max level must be non-negative: " + maxLevel);
    }
    this.internalId = internalId;
    this.maxLevel = maxLevel;
    this.neighbors = new int[maxLevel + 1][];
    for (int l = 0; l <= maxLevel; l++) {
      this.neighbors[l] = new int[0];
    }
  }

  public int internalId() {
    return internalId;
  }

  public int maxLevel() {
    return maxLevel;
  }

  /**
   * Returns the neighbor IDs at the specified layer.
   *
   * @param layer the layer index (0 to maxLevel inclusive)
   * @return array of internal IDs of neighbors at that layer
   */
  public int[] getNeighbors(int layer) {
    checkLayerBounds(layer);
    return neighbors[layer];
  }

  /**
   * Replaces the neighbor list at the specified layer.
   *
   * @param layer the layer index
   * @param newNeighbors the new neighbor array (will be stored directly, not copied)
   */
  public void setNeighbors(int layer, int[] newNeighbors) {
    checkLayerBounds(layer);
    this.neighbors[layer] = newNeighbors;
  }

  /** Returns the number of neighbors (degree) at the specified layer. */
  public int degree(int layer) {
    checkLayerBounds(layer);
    return neighbors[layer].length;
  }

  /** Checks whether this node has the given neighbor at the specified layer. */
  public boolean hasNeighbor(int layer, int neighborId) {
    int[] layerNeighbors = getNeighbors(layer);
    for (int n : layerNeighbors) {
      if (n == neighborId) {
        return true;
      }
    }
    return false;
  }

  /**
   * Adds a neighbor to this node's adjacency list at the specified layer. Does not check for
   * duplicates (caller is responsible).
   *
   * @param layer the layer index
   * @param neighborId internal ID of the neighbor to add
   */
  public void addNeighbor(int layer, int neighborId) {
    checkLayerBounds(layer);
    int[] old = neighbors[layer];
    int[] expanded = Arrays.copyOf(old, old.length + 1);
    expanded[old.length] = neighborId;
    neighbors[layer] = expanded;
  }

  /**
   * Removes a neighbor from this node's adjacency list at the specified layer, if present.
   *
   * @param layer the layer index
   * @param neighborId internal ID of the neighbor to remove
   */
  public void removeNeighbor(int layer, int neighborId) {
    checkLayerBounds(layer);
    int[] old = neighbors[layer];
    int index = -1;
    for (int i = 0; i < old.length; i++) {
      if (old[i] == neighborId) {
        index = i;
        break;
      }
    }
    if (index == -1) {
      return;
    }
    int[] updated = new int[old.length - 1];
    System.arraycopy(old, 0, updated, 0, index);
    System.arraycopy(old, index + 1, updated, index, old.length - index - 1);
    neighbors[layer] = updated;
  }

  private void checkLayerBounds(int layer) {
    if (layer < 0 || layer > maxLevel) {
      throw new IndexOutOfBoundsException(
          "Layer "
              + layer
              + " out of bounds for node "
              + internalId
              + " (maxLevel="
              + maxLevel
              + ")");
    }
  }
}
