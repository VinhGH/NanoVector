package com.nanovector.core.offheap;

import com.nanovector.core.hnsw.EpochVisitedSet;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.hnsw.NeighborSelector;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;

/**
 * High-performance off-heap HNSW multi-layer graph topology engine.
 *
 * <p>Delegates all memory storage and neighbor adjacency indexing to {@link OffHeapGraphLayout},
 * avoiding any Java heap objects for nodes, arrays, or edge pointers.
 *
 * <p>Preserves 100% behavioral equivalence, invariant protection, degree bounding, and tie-breaking
 * logic identical to the on-heap {@link HnswGraph}.
 */
public final class OffHeapHnswGraph implements AutoCloseable {

  private final HnswConfig config;
  private final OffHeapGraphLayout layout;
  private int entryPointId;
  private int maxLevel;

  public OffHeapHnswGraph(HnswConfig config) {
    this(config, new OffHeapGraphLayout(config));
  }

  public OffHeapHnswGraph(HnswConfig config, int initialCapacity) {
    this(config, new OffHeapGraphLayout(config, initialCapacity));
  }

  public OffHeapHnswGraph(HnswConfig config, OffHeapGraphLayout layout) {
    this.config = Objects.requireNonNull(config, "HnswConfig must not be null");
    this.layout = Objects.requireNonNull(layout, "OffHeapGraphLayout must not be null");
    this.entryPointId = -1;
    this.maxLevel = -1;
  }

  /**
   * Adds a new node to the off-heap graph. If this is the first node, it becomes the initial entry
   * point.
   */
  public void addNode(int internalId, int maxLevel) {
    if (internalId != layout.nodeCount()) {
      throw new IllegalArgumentException(
          "Node internalId must be " + layout.nodeCount() + ", but got: " + internalId);
    }
    layout.addNode(internalId, maxLevel);

    if (entryPointId == -1) {
      entryPointId = internalId;
      this.maxLevel = maxLevel;
    }
  }

  /** Updates the graph's entry point and maximum level. */
  public void setEntryPoint(int entryPointId, int maxLevel) {
    if (entryPointId < 0 || entryPointId >= layout.nodeCount()) {
      throw new IndexOutOfBoundsException(
          "Entry point ID "
              + entryPointId
              + " out of bounds, graph has "
              + layout.nodeCount()
              + " nodes");
    }
    this.entryPointId = entryPointId;
    this.maxLevel = maxLevel;
  }

  // ── Graph Traversal ─────────────────────────────────────────────────

  /**
   * Searches within a single layer of the graph to find the {@code ef} closest nodes to the query
   * (Algorithm 2 of Malkov &amp; Yashunin 2018).
   */
  public List<NeighborSelector.Candidate> searchLayer(
      HnswGraph.DistanceToQuery distanceToQuery,
      int[] entryPoints,
      int ef,
      int layer,
      EpochVisitedSet visitedSet) {
    PriorityQueue<NeighborSelector.Candidate> candidates = new PriorityQueue<>();
    PriorityQueue<NeighborSelector.Candidate> results =
        new PriorityQueue<>(Comparator.reverseOrder());

    for (int ep : entryPoints) {
      if (!visitedSet.isVisited(ep)) {
        visitedSet.markVisited(ep);
        float dist = distanceToQuery.distance(ep);
        NeighborSelector.Candidate c = new NeighborSelector.Candidate(ep, dist);
        candidates.add(c);
        results.add(c);
      }
    }

    while (!candidates.isEmpty()) {
      NeighborSelector.Candidate closest = candidates.poll();

      assert results.peek() != null;
      if (closest.distance() > results.peek().distance()) {
        break;
      }

      int[] neighbors = layout.getNeighbors(closest.id(), layer);

      for (int neighborId : neighbors) {
        if (visitedSet.isVisited(neighborId)) {
          continue;
        }
        visitedSet.markVisited(neighborId);

        float neighborDist = distanceToQuery.distance(neighborId);

        assert results.peek() != null;
        if (results.size() < ef || neighborDist < results.peek().distance()) {
          NeighborSelector.Candidate neighborCandidate =
              new NeighborSelector.Candidate(neighborId, neighborDist);
          candidates.add(neighborCandidate);
          results.add(neighborCandidate);

          if (results.size() > ef) {
            results.poll();
          }
        }
      }
    }

    List<NeighborSelector.Candidate> sorted = new ArrayList<>(results);
    sorted.sort(null);
    return sorted;
  }

  /**
   * Greedy 1-nearest-neighbor descent within a single layer. Uses direct index access via {@link
   * OffHeapGraphLayout#getNeighbor(int, int, int)} to avoid heap allocation.
   */
  public int greedyClosest(
      HnswGraph.DistanceToQuery distanceToQuery, int currentBestId, int layer) {
    int bestId = currentBestId;
    float bestDist = distanceToQuery.distance(bestId);
    boolean improved = true;

    while (improved) {
      improved = false;
      int deg = layout.degree(bestId, layer);

      for (int i = 0; i < deg; i++) {
        int neighborId = layout.getNeighbor(bestId, layer, i);
        float neighborDist = distanceToQuery.distance(neighborId);
        if (neighborDist < bestDist) {
          bestId = neighborId;
          bestDist = neighborDist;
          improved = true;
        }
      }
    }
    return bestId;
  }

  // ── Edge Management ─────────────────────────────────────────────────

  /**
   * Creates a bidirectional edge between two nodes at the specified layer, with pruning if degree
   * exceeds the maximum allowed.
   */
  public void connect(
      int nodeA, int nodeB, int layer, NeighborSelector.NodeDistanceEvaluator evaluator) {
    if (nodeA == nodeB) {
      return;
    }

    if (!layout.hasNeighbor(nodeA, layer, nodeB)) {
      layout.addNeighbor(nodeA, layer, nodeB);
    }
    if (!layout.hasNeighbor(nodeB, layer, nodeA)) {
      layout.addNeighbor(nodeB, layer, nodeA);
    }

    int maxDegree = maxDegreeForLayer(layer);

    if (layout.degree(nodeA, layer) > maxDegree) {
      prune(nodeA, layer, maxDegree, evaluator);
    }
    if (layout.degree(nodeB, layer) > maxDegree) {
      prune(nodeB, layer, maxDegree, evaluator);
    }
  }

  /** Removes a bidirectional edge between two nodes at the specified layer. */
  public void removeBidirectional(int nodeA, int nodeB, int layer) {
    layout.removeNeighbor(nodeA, layer, nodeB);
    layout.removeNeighbor(nodeB, layer, nodeA);
  }

  /**
   * Prunes a node's neighbor list at a given layer to at most {@code maxDegree} neighbors using the
   * {@link NeighborSelector} heuristic with fallback and Layer 0 invariant preservation.
   */
  public void prune(
      int nodeId, int layer, int maxDegree, NeighborSelector.NodeDistanceEvaluator evaluator) {
    int[] currentNeighbors = layout.getNeighbors(nodeId, layer);
    if (currentNeighbors.length <= maxDegree) {
      return;
    }

    List<NeighborSelector.Candidate> candidates = new ArrayList<>(currentNeighbors.length);
    for (int neighborId : currentNeighbors) {
      float dist = evaluator.distance(nodeId, neighborId);
      candidates.add(new NeighborSelector.Candidate(neighborId, dist));
    }

    int[] pruned = NeighborSelector.selectNeighbors(candidates, maxDegree, evaluator);

    boolean[] kept = new boolean[layout.nodeCount()];
    for (int id : pruned) {
      kept[id] = true;
    }

    // Invariant protection for Layer 0: prevent creating isolated nodes (degree == 0).
    if (layer == 0) {
      for (int oldNeighborId : currentNeighbors) {
        if (!kept[oldNeighborId]) {
          if (layout.degree(oldNeighborId, 0) <= 1) {
            for (int i = pruned.length - 1; i >= 0; i--) {
              int candId = pruned[i];
              if (candId != oldNeighborId && layout.degree(candId, 0) > 1) {
                kept[candId] = false;
                kept[oldNeighborId] = true;
                pruned[i] = oldNeighborId;
                break;
              }
            }
          }
        }
      }
    }

    for (int oldNeighborId : currentNeighbors) {
      if (!kept[oldNeighborId]) {
        layout.removeNeighbor(oldNeighborId, layer, nodeId);
      }
    }

    layout.setNeighbors(nodeId, layer, pruned);
  }

  /** Converts the off-heap node state at {@code internalId} into an on-heap {@link HnswNode}. */
  public HnswNode toNode(int internalId) {
    int maxL = layout.maxLevel(internalId);
    HnswNode node = new HnswNode(internalId, maxL);
    for (int l = 0; l <= maxL; l++) {
      node.setNeighbors(l, layout.getNeighbors(internalId, l));
    }
    return node;
  }

  // ── Accessors ───────────────────────────────────────────────────────

  public int maxDegreeForLayer(int layer) {
    return layer == 0 ? config.m0() : config.m();
  }

  public int entryPointId() {
    return entryPointId;
  }

  public int maxLevel() {
    return maxLevel;
  }

  public int size() {
    return layout.nodeCount();
  }

  public boolean isEmpty() {
    return layout.nodeCount() == 0;
  }

  public HnswConfig config() {
    return config;
  }

  public OffHeapGraphLayout layout() {
    return layout;
  }

  public int[] getNeighbors(int internalId, int layer) {
    return layout.getNeighbors(internalId, layer);
  }

  public int degree(int internalId, int layer) {
    return layout.degree(internalId, layer);
  }

  public boolean hasNeighbor(int internalId, int layer, int neighborId) {
    return layout.hasNeighbor(internalId, layer, neighborId);
  }

  @Override
  public void close() {
    layout.close();
  }
}
