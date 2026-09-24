package com.nanovector.core.hnsw;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class SearchLayerTest {

  /**
   * Builds a small 1D graph for testing searchLayer and greedyClosest.
   *
   * <p>Layout (layer 0 only): Node 0 (pos=0) -- Node 1 (pos=10) -- Node 2 (pos=20) -- Node 3
   * (pos=30) -- Node 4 (pos=40)
   *
   * <p>Each node is connected to its immediate neighbors forming a chain.
   */
  private static final float[] POSITIONS = {0f, 10f, 20f, 30f, 40f};

  private static final HnswConfig CONFIG = new HnswConfig(4, 8, 100, 50, 1.0 / Math.log(4), 42L);

  private HnswGraph buildChainGraph() {
    HnswGraph graph = new HnswGraph(CONFIG);
    for (int i = 0; i < POSITIONS.length; i++) {
      graph.addNode(new HnswNode(i, 0));
    }
    // Chain: 0-1, 1-2, 2-3, 3-4
    NeighborSelector.NodeDistanceEvaluator nodeEval =
        (a, b) -> Math.abs(POSITIONS[a] - POSITIONS[b]);
    for (int i = 0; i < POSITIONS.length - 1; i++) {
      graph.connect(i, i + 1, 0, nodeEval);
    }
    return graph;
  }

  private HnswGraph.DistanceToQuery queryAt(float queryPos) {
    return (internalId) -> Math.abs(POSITIONS[internalId] - queryPos);
  }

  @Test
  @DisplayName(
      "searchLayer: Finds closest nodes in a chain graph starting from a distant entry point")
  void testSearchLayerFindsClosestNodes() {
    HnswGraph graph = buildChainGraph();
    EpochVisitedSet visited = new EpochVisitedSet(POSITIONS.length);

    // Query at position 35, starting from node 0 (position 0, far away)
    // Should traverse the chain and find nodes 3 (pos=30, dist=5) and 4 (pos=40, dist=5)
    visited.nextEpoch();
    List<NeighborSelector.Candidate> results =
        graph.searchLayer(
            queryAt(35f),
            new int[] {0}, // start from the farthest node
            3, // ef = 3
            0, // layer 0
            visited);

    assertThat(results).isNotEmpty();
    // The two closest nodes to 35 are: 3 (dist=5) and 4 (dist=5)
    assertThat(results.get(0).id()).isIn(3, 4);
    assertThat(results.get(0).distance()).isEqualTo(5.0f);
  }

  @Test
  @DisplayName("searchLayer: Respects ef bound on result size")
  void testSearchLayerRespectsEfBound() {
    HnswGraph graph = buildChainGraph();
    EpochVisitedSet visited = new EpochVisitedSet(POSITIONS.length);

    visited.nextEpoch();
    List<NeighborSelector.Candidate> results =
        graph.searchLayer(
            queryAt(20f), // query right at node 2
            new int[] {0},
            2, // ef = 2
            0,
            visited);

    assertThat(results).hasSizeLessThanOrEqualTo(2);
    // Node 2 at distance 0 must be the closest
    assertThat(results.get(0).id()).isEqualTo(2);
    assertThat(results.get(0).distance()).isEqualTo(0.0f);
  }

  @Test
  @DisplayName("searchLayer: Returns results sorted ascending by distance")
  void testSearchLayerReturnsSortedResults() {
    HnswGraph graph = buildChainGraph();
    EpochVisitedSet visited = new EpochVisitedSet(POSITIONS.length);

    visited.nextEpoch();
    List<NeighborSelector.Candidate> results =
        graph.searchLayer(
            queryAt(15f),
            new int[] {0},
            5, // ef = 5
            0,
            visited);

    // Verify ascending distance order
    for (int i = 1; i < results.size(); i++) {
      assertThat(results.get(i).distance()).isGreaterThanOrEqualTo(results.get(i - 1).distance());
    }
  }

  @Test
  @DisplayName("greedyClosest: Traverses chain to find the nearest node")
  void testGreedyClosestFindsNearest() {
    HnswGraph graph = buildChainGraph();

    // Query at position 25, start at node 0 (position 0)
    int closest = graph.greedyClosest(queryAt(25f), 0, 0);

    // Should traverse 0 -> 1 -> 2 -> 3 (pos=30, dist=5) or stop at 2 (pos=20, dist=5)
    // Both are equidistant. Greedy will find 3 since 30 is checked after 20 and improves.
    assertThat(closest).isIn(2, 3);
  }

  @Test
  @DisplayName("greedyClosest: Stays at entry point if it is already closest")
  void testGreedyClosestStaysAtOptimal() {
    HnswGraph graph = buildChainGraph();

    // Query exactly at position 20 (node 2), start at node 2
    int closest = graph.greedyClosest(queryAt(20f), 2, 0);

    assertThat(closest).isEqualTo(2);
  }

  @Test
  @DisplayName(
      "Multi-layer routing: greedyClosest descends through upper layers to find entry for layer 0")
  void testMultiLayerGreedyRouting() {
    HnswGraph graph = new HnswGraph(CONFIG);

    // Create a 2-layer graph:
    // Layer 1: Node 0 (pos=0) -- Node 3 (pos=30)
    // Layer 0: All nodes 0-4 chained
    float[] positions = {0f, 10f, 20f, 30f, 40f};
    graph.addNode(new HnswNode(0, 1)); // exists on layers 0 and 1
    graph.addNode(new HnswNode(1, 0)); // only layer 0
    graph.addNode(new HnswNode(2, 0)); // only layer 0
    graph.addNode(new HnswNode(3, 1)); // exists on layers 0 and 1
    graph.addNode(new HnswNode(4, 0)); // only layer 0

    NeighborSelector.NodeDistanceEvaluator nodeEval =
        (a, b) -> Math.abs(positions[a] - positions[b]);

    // Layer 1 edge: 0 -- 3
    graph.connect(0, 3, 1, nodeEval);

    // Layer 0 chain: 0-1-2-3-4
    for (int i = 0; i < positions.length - 1; i++) {
      graph.connect(i, i + 1, 0, nodeEval);
    }

    HnswGraph.DistanceToQuery queryDist = (id) -> Math.abs(positions[id] - 35f);

    // Step 1: Greedy routing on layer 1, starting from node 0
    int bestAfterLayer1 = graph.greedyClosest(queryDist, 0, 1);
    assertThat(bestAfterLayer1).isEqualTo(3); // Node 3 (pos=30) is closer to 35 than node 0

    // Step 2: searchLayer on layer 0, starting from the result of layer 1
    EpochVisitedSet visited = new EpochVisitedSet(positions.length);
    visited.nextEpoch();
    List<NeighborSelector.Candidate> results =
        graph.searchLayer(queryDist, new int[] {bestAfterLayer1}, 3, 0, visited);

    // Node 4 (pos=40, dist=5) and node 3 (pos=30, dist=5) are closest
    assertThat(results.get(0).distance()).isEqualTo(5.0f);
    assertThat(results.get(0).id()).isIn(3, 4);
  }
}
