package com.nanovector.core.hnsw;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class HnswGraphTest {

    private static final HnswConfig CONFIG = new HnswConfig(
            4,   // M = 4 (small for easy test verification)
            8,   // M0 = 8
            100,
            50,
            1.0 / Math.log(4),
            42L
    );

    // Simple evaluator: distance = abs(a - b) to simulate a 1D scenario
    private static final NeighborSelector.NodeDistanceEvaluator EVALUATOR =
            (a, b) -> Math.abs(a - b);

    @Test
    @DisplayName("HnswNode: Created with correct level and empty neighbor lists")
    void testNodeCreation() {
        HnswNode node = new HnswNode(0, 3);

        assertThat(node.internalId()).isEqualTo(0);
        assertThat(node.maxLevel()).isEqualTo(3);
        for (int l = 0; l <= 3; l++) {
            assertThat(node.getNeighbors(l)).isEmpty();
            assertThat(node.degree(l)).isZero();
        }
    }

    @Test
    @DisplayName("HnswNode: addNeighbor and hasNeighbor work correctly")
    void testNodeAddAndHasNeighbor() {
        HnswNode node = new HnswNode(0, 1);
        node.addNeighbor(0, 5);
        node.addNeighbor(0, 10);

        assertThat(node.degree(0)).isEqualTo(2);
        assertThat(node.hasNeighbor(0, 5)).isTrue();
        assertThat(node.hasNeighbor(0, 10)).isTrue();
        assertThat(node.hasNeighbor(0, 99)).isFalse();
        assertThat(node.hasNeighbor(1, 5)).isFalse(); // Different layer
    }

    @Test
    @DisplayName("HnswNode: Layer bounds checking prevents out-of-range access")
    void testNodeLayerBoundsChecking() {
        HnswNode node = new HnswNode(0, 2);

        assertThatThrownBy(() -> node.getNeighbors(3))
                .isInstanceOf(IndexOutOfBoundsException.class);
        assertThatThrownBy(() -> node.getNeighbors(-1))
                .isInstanceOf(IndexOutOfBoundsException.class);
    }

    @Test
    @DisplayName("HnswGraph: First node becomes entry point at its level")
    void testFirstNodeBecomesEntryPoint() {
        HnswGraph graph = new HnswGraph(CONFIG);

        assertThat(graph.isEmpty()).isTrue();
        assertThat(graph.entryPointId()).isEqualTo(-1);

        graph.addNode(new HnswNode(0, 2));

        assertThat(graph.isEmpty()).isFalse();
        assertThat(graph.size()).isEqualTo(1);
        assertThat(graph.entryPointId()).isEqualTo(0);
        assertThat(graph.maxLevel()).isEqualTo(2);
    }

    @Test
    @DisplayName("HnswGraph: Entry point updates only when a higher-level node is added")
    void testEntryPointUpdatesOnHigherLevel() {
        HnswGraph graph = new HnswGraph(CONFIG);
        graph.addNode(new HnswNode(0, 2));
        graph.addNode(new HnswNode(1, 1)); // lower level -> no update

        assertThat(graph.entryPointId()).isEqualTo(0);
        assertThat(graph.maxLevel()).isEqualTo(2);

        graph.addNode(new HnswNode(2, 5)); // higher level -> update

        assertThat(graph.entryPointId()).isEqualTo(2);
        assertThat(graph.maxLevel()).isEqualTo(5);
    }

    @Test
    @DisplayName("HnswGraph: connect creates bidirectional edges")
    void testBidirectionalConnect() {
        HnswGraph graph = new HnswGraph(CONFIG);
        graph.addNode(new HnswNode(0, 0));
        graph.addNode(new HnswNode(1, 0));
        graph.addNode(new HnswNode(2, 0));

        graph.connect(0, 1, 0, EVALUATOR);
        graph.connect(0, 2, 0, EVALUATOR);

        // Verify bidirectional
        assertThat(graph.getNode(0).hasNeighbor(0, 1)).isTrue();
        assertThat(graph.getNode(1).hasNeighbor(0, 0)).isTrue();
        assertThat(graph.getNode(0).hasNeighbor(0, 2)).isTrue();
        assertThat(graph.getNode(2).hasNeighbor(0, 0)).isTrue();

        // Node 1 and Node 2 are NOT connected
        assertThat(graph.getNode(1).hasNeighbor(0, 2)).isFalse();
    }

    @Test
    @DisplayName("HnswGraph: connect does not create self-loops")
    void testNoSelfLoops() {
        HnswGraph graph = new HnswGraph(CONFIG);
        graph.addNode(new HnswNode(0, 0));

        graph.connect(0, 0, 0, EVALUATOR);

        assertThat(graph.getNode(0).degree(0)).isZero();
    }

    @Test
    @DisplayName("HnswGraph: connect does not create duplicate edges")
    void testNoDuplicateEdges() {
        HnswGraph graph = new HnswGraph(CONFIG);
        graph.addNode(new HnswNode(0, 0));
        graph.addNode(new HnswNode(1, 0));

        graph.connect(0, 1, 0, EVALUATOR);
        graph.connect(0, 1, 0, EVALUATOR); // second call
        graph.connect(1, 0, 0, EVALUATOR); // reverse call

        assertThat(graph.getNode(0).degree(0)).isEqualTo(1);
        assertThat(graph.getNode(1).degree(0)).isEqualTo(1);
    }

    @Test
    @DisplayName("HnswGraph: prune enforces degree constraint M at layer > 0")
    void testPruneEnforcesDegreeConstraint() {
        // M = 4, so max degree at layer 1 = 4
        HnswGraph graph = new HnswGraph(CONFIG);
        for (int i = 0; i <= 6; i++) {
            graph.addNode(new HnswNode(i, 1));
        }

        // Manually connect node 0 to nodes 1..6 at layer 1 (6 neighbors > M=4)
        HnswNode node0 = graph.getNode(0);
        for (int i = 1; i <= 6; i++) {
            node0.addNeighbor(1, i);
        }
        assertThat(node0.degree(1)).isEqualTo(6);

        // Prune should reduce to M = 4
        graph.prune(node0, 1, CONFIG.m(), EVALUATOR);

        assertThat(node0.degree(1)).isLessThanOrEqualTo(4);
    }

    @Test
    @DisplayName("HnswGraph: connect triggers automatic pruning when degree exceeds limit")
    void testConnectTriggersAutoPruning() {
        // M = 4 at layer > 0
        HnswGraph graph = new HnswGraph(CONFIG);
        for (int i = 0; i <= 5; i++) {
            graph.addNode(new HnswNode(i, 1));
        }

        // Connect node 0 to nodes 1, 2, 3, 4 at layer 1 (degree = 4 = M, exactly at limit)
        for (int i = 1; i <= 4; i++) {
            graph.connect(0, i, 1, EVALUATOR);
        }
        assertThat(graph.getNode(0).degree(1)).isEqualTo(4);

        // One more connection pushes to 5 > M=4, should trigger auto-prune
        graph.connect(0, 5, 1, EVALUATOR);

        assertThat(graph.getNode(0).degree(1)).isLessThanOrEqualTo(4);
    }
}
