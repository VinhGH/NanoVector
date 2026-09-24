package com.nanovector.core.hnsw;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Multi-layer HNSW graph managing nodes, entry point, edge connections, and pruning.
 * <p>
 * Responsibilities:
 * <ul>
 *   <li>Maintains the global {@code entryPoint} at the current maximum level ({@code maxLevel}).</li>
 *   <li>Stores all {@link HnswNode}s indexed by their internal ID.</li>
 *   <li>Provides bidirectional {@code connect} and {@code prune} operations that
 *       preserve the Graph Invariants (degree constraints, symmetric edges).</li>
 * </ul>
 * <p>
 * <b>Design note:</b> {@code HnswGraph} does not hold vector data or compute distances.
 * Distance computations are injected via {@link NeighborSelector.NodeDistanceEvaluator}
 * by the orchestrating {@code HnswIndex}.
 */
public final class HnswGraph {

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

        if (entryPointId == -1 || node.maxLevel() > maxLevel) {
            entryPointId = node.internalId();
            maxLevel = node.maxLevel();
        }
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

    /**
     * Returns the maximum degree for the given layer.
     * Layer 0 uses {@code M0} (= 2M), layers > 0 use {@code M}.
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
