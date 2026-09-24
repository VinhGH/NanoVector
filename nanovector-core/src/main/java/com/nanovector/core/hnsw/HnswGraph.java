package com.nanovector.core.hnsw;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;

/**
 * Multi-layer HNSW graph managing nodes, entry point, edge connections, pruning,
 * and graph traversal (greedy routing &amp; searchLayer).
 * <p>
 * Responsibilities:
 * <ul>
 *   <li>Maintains the global {@code entryPoint} at the current maximum level ({@code maxLevel}).</li>
 *   <li>Stores all {@link HnswNode}s indexed by their internal ID.</li>
 *   <li>Provides bidirectional {@code connect} and {@code prune} operations that
 *       preserve the Graph Invariants (degree constraints, symmetric edges).</li>
 *   <li>Implements {@code searchLayer} (Algorithm 2) and {@code greedyClosest} for
 *       multi-layer greedy routing.</li>
 * </ul>
 * <p>
 * <b>Design note:</b> {@code HnswGraph} does not hold vector data or compute distances.
 * Distance computations are injected via functional interfaces
 * ({@link DistanceToQuery}, {@link NeighborSelector.NodeDistanceEvaluator})
 * by the orchestrating {@code HnswIndex}.
 */
public final class HnswGraph {

    /**
     * Functional interface for computing the distance from a query to a graph node.
     * <p>
     * The orchestrating {@code HnswIndex} creates this by closing over the query vector,
     * {@code VectorStorage}, and {@code DistanceCalculator}, keeping {@code HnswGraph}
     * fully decoupled from vector data.
     */
    @FunctionalInterface
    public interface DistanceToQuery {
        /**
         * Computes the distance from the current query to the node at the given internal ID.
         *
         * @param internalId the node's internal index
         * @return distance value (smaller is closer, per the distance minimization contract)
         */
        float distance(int internalId);
    }

    private final HnswConfig config;
    private final List<HnswNode> nodes;
    private int entryPointId;
    private int maxLevel;

    public HnswGraph(HnswConfig config) {
        this.config = Objects.requireNonNull(config, "HnswConfig must not be null");
        this.nodes = new ArrayList<>();
        this.entryPointId = -1;
        this.maxLevel = -1;
    }

    /**
     * Adds a new node to the graph.
     * If this is the first node added, it becomes the initial entry point.
     * Subsequent entry point updates are managed via {@link #setEntryPoint(int, int)}.
     *
     * @param node the node to add (its internalId must equal the current node count)
     */
    public void addNode(HnswNode node) {
        Objects.requireNonNull(node, "Node must not be null");
        if (node.internalId() != nodes.size()) {
            throw new IllegalArgumentException(
                    "Node internalId must be " + nodes.size() + ", but got: " + node.internalId());
        }
        nodes.add(node);

        if (entryPointId == -1) {
            entryPointId = node.internalId();
            maxLevel = node.maxLevel();
        }
    }

    /**
     * Updates the entry point and maximum level of the graph.
     *
     * @param entryPointId internal ID of the new entry point
     * @param maxLevel     the maximum level of the new entry point
     */
    public void setEntryPoint(int entryPointId, int maxLevel) {
        if (entryPointId < 0 || entryPointId >= nodes.size()) {
            throw new IndexOutOfBoundsException(
                    "Entry point ID " + entryPointId + " out of bounds, graph has " + nodes.size() + " nodes");
        }
        this.entryPointId = entryPointId;
        this.maxLevel = maxLevel;
    }

    /**
     * Returns the node at the given internal index.
     */
    public HnswNode getNode(int internalId) {
        if (internalId < 0 || internalId >= nodes.size()) {
            throw new IndexOutOfBoundsException(
                    "Internal ID " + internalId + " out of bounds, graph has " + nodes.size() + " nodes");
        }
        return nodes.get(internalId);
    }

    // ── Graph Traversal ─────────────────────────────────────────────────

    /**
     * Searches within a single layer of the graph to find the {@code ef} closest nodes
     * to the query (Algorithm 2 of Malkov &amp; Yashunin 2018).
     * <p>
     * Uses a Min-Heap (candidate queue) to explore the closest unvisited candidate,
     * and a Max-Heap (dynamic result set) bounded to {@code ef} to track the best results.
     *
     * @param distanceToQuery distance function from query to any node
     * @param entryPoints     starting node IDs for the search on this layer
     * @param ef              size of the dynamic candidate list (controls accuracy vs speed)
     * @param layer           the layer to search within
     * @param visitedSet      epoch-based visited tracking (caller must call {@code nextEpoch()} beforehand)
     * @return list of {@link NeighborSelector.Candidate} sorted ascending by distance
     */
    public List<NeighborSelector.Candidate> searchLayer(
            DistanceToQuery distanceToQuery,
            int[] entryPoints,
            int ef,
            int layer,
            EpochVisitedSet visitedSet
    ) {
        // Min-Heap: candidates to explore (closest first)
        PriorityQueue<NeighborSelector.Candidate> candidates = new PriorityQueue<>();
        // Max-Heap: dynamic result set (farthest first, bounded to ef)
        PriorityQueue<NeighborSelector.Candidate> results =
                new PriorityQueue<>(Comparator.reverseOrder());

        // Seed with entry points
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
            // Extract closest unexplored candidate
            NeighborSelector.Candidate closest = candidates.poll();

            // If closest candidate is farther than the worst result, stop
            // (no remaining candidate can improve the result set)
            assert results.peek() != null;
            if (closest.distance() > results.peek().distance()) {
                break;
            }

            // Explore neighbors of the closest candidate at this layer
            HnswNode closestNode = getNode(closest.id());
            int[] neighbors = closestNode.getNeighbors(layer);

            for (int neighborId : neighbors) {
                if (visitedSet.isVisited(neighborId)) {
                    continue;
                }
                visitedSet.markVisited(neighborId);

                float neighborDist = distanceToQuery.distance(neighborId);

                // Add to results if result set is not full, or if this neighbor is closer
                // than the current worst result
                assert results.peek() != null;
                if (results.size() < ef || neighborDist < results.peek().distance()) {
                    NeighborSelector.Candidate neighborCandidate =
                            new NeighborSelector.Candidate(neighborId, neighborDist);
                    candidates.add(neighborCandidate);
                    results.add(neighborCandidate);

                    // Evict the farthest result if result set exceeds ef
                    if (results.size() > ef) {
                        results.poll();
                    }
                }
            }
        }

        // Convert results to sorted list (ascending by distance)
        List<NeighborSelector.Candidate> sorted = new ArrayList<>(results);
        sorted.sort(null); // Natural order: ascending by distance, then by id
        return sorted;
    }

    /**
     * Greedy 1-nearest-neighbor descent within a single layer.
     * Used during the upper-layer routing phase (layers {@code Lmax} down to {@code targetLevel + 1})
     * where only the single closest node is tracked.
     *
     * @param distanceToQuery distance function from query to any node
     * @param currentBestId   current closest node ID entering this layer
     * @param layer           the layer to descend through
     * @return internal ID of the closest node found at this layer
     */
    public int greedyClosest(
            DistanceToQuery distanceToQuery,
            int currentBestId,
            int layer
    ) {
        int bestId = currentBestId;
        float bestDist = distanceToQuery.distance(bestId);
        boolean improved = true;

        while (improved) {
            improved = false;
            HnswNode bestNode = getNode(bestId);
            int[] neighbors = bestNode.getNeighbors(layer);

            for (int neighborId : neighbors) {
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
     * Creates a bidirectional edge between two nodes at the specified layer.
     * If either node's degree exceeds the maximum allowed, calls {@code prune} on that node.
     *
     * @param nodeA    internal ID of first node
     * @param nodeB    internal ID of second node
     * @param layer    the layer at which to connect
     * @param evaluator distance evaluator for pruning decisions
     */
    public void connect(int nodeA, int nodeB, int layer,
                        NeighborSelector.NodeDistanceEvaluator evaluator) {
        if (nodeA == nodeB) {
            return; // No self-loops
        }

        HnswNode a = getNode(nodeA);
        HnswNode b = getNode(nodeB);

        if (!a.hasNeighbor(layer, nodeB)) {
            a.addNeighbor(layer, nodeB);
        }
        if (!b.hasNeighbor(layer, nodeA)) {
            b.addNeighbor(layer, nodeA);
        }

        int maxDegree = maxDegreeForLayer(layer);

        if (a.degree(layer) > maxDegree) {
            prune(a, layer, maxDegree, evaluator);
        }
        if (b.degree(layer) > maxDegree) {
            prune(b, layer, maxDegree, evaluator);
        }
    }

    /**
     * Prunes a node's neighbor list at a given layer to at most {@code maxDegree} neighbors
     * using the {@link NeighborSelector} heuristic with fallback.
     *
     * @param node       the node whose neighbors to prune
     * @param layer      the layer at which to prune
     * @param maxDegree  maximum allowed neighbors
     * @param evaluator  distance evaluator for neighbor selection
     */
    public void prune(HnswNode node, int layer, int maxDegree,
                      NeighborSelector.NodeDistanceEvaluator evaluator) {
        int[] currentNeighbors = node.getNeighbors(layer);
        if (currentNeighbors.length <= maxDegree) {
            return;
        }

        int nodeId = node.internalId();
        List<NeighborSelector.Candidate> candidates = new ArrayList<>(currentNeighbors.length);
        for (int neighborId : currentNeighbors) {
            float dist = evaluator.distance(nodeId, neighborId);
            candidates.add(new NeighborSelector.Candidate(neighborId, dist));
        }

        int[] pruned = NeighborSelector.selectNeighbors(candidates, maxDegree, evaluator);
        node.setNeighbors(layer, pruned);
    }

    // ── Accessors ───────────────────────────────────────────────────────

    /**
     * Returns the maximum degree for the given layer.
     * Layer 0 uses {@code M0} (= 2M), layers &gt; 0 use {@code M}.
     */
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
        return nodes.size();
    }

    public boolean isEmpty() {
        return nodes.isEmpty();
    }

    public HnswConfig config() {
        return config;
    }
}
