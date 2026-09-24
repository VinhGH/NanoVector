package com.nanovector.core.hnsw;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.model.SearchResult;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class HnswGraphInvariantTest {

    private static final int DIMENSION = 16;
    private static final int NUM_VECTORS = 200;
    private static final long SEED = 42L;

    private static float[] randomVector(Random rng, int dim) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) {
            v[i] = rng.nextFloat() * 2f - 1f;
        }
        return v;
    }

    private HnswIndex buildIndex(long seed, int n) {
        HnswConfig config = HnswConfig.defaultConfig().withSeed(seed);
        HnswIndex index = new HnswIndex(DIMENSION, DistanceMetric.EUCLIDEAN, config);
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            index.insert(i, randomVector(rng, DIMENSION));
        }
        return index;
    }

    @Test
    @DisplayName("Invariant 1: Degree constraints - layer 0 degree <= M0, layer > 0 degree <= M")
    void testDegreeConstraints() {
        HnswIndex index = buildIndex(SEED, NUM_VECTORS);
        HnswGraph graph = index.graph();
        HnswConfig config = index.config();

        for (int i = 0; i < graph.size(); i++) {
            HnswNode node = graph.getNode(i);
            // Layer 0 constraint: degree <= M0
            assertThat(node.degree(0))
                    .as("Node %d degree at layer 0 must be <= M0 (%d)", i, config.m0())
                    .isLessThanOrEqualTo(config.m0());

            // Layer > 0 constraint: degree <= M
            for (int l = 1; l <= node.maxLevel(); l++) {
                assertThat(node.degree(l))
                        .as("Node %d degree at layer %d must be <= M (%d)", i, l, config.m())
                        .isLessThanOrEqualTo(config.m());
            }
        }
    }

    @Test
    @DisplayName("Invariant 2: Layer 0 connectivity - BFS from node 0 visits 100% of nodes")
    void testLayer0ConnectivityViaBFS() {
        HnswIndex index = buildIndex(SEED, NUM_VECTORS);
        HnswGraph graph = index.graph();

        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue = new ArrayDeque<>();

        queue.add(0);
        visited.add(0);

        while (!queue.isEmpty()) {
            int current = queue.poll();
            for (int neighbor : graph.getNode(current).getNeighbors(0)) {
                if (visited.add(neighbor)) {
                    queue.add(neighbor);
                }
            }
        }

        assertThat(visited)
                .as("BFS on layer 0 should reach all %d nodes", NUM_VECTORS)
                .hasSize(NUM_VECTORS);
    }

    @Test
    @DisplayName("Invariant 3: Global Entry Point is at graph maxLevel")
    void testEntryPointAtMaxLevel() {
        HnswIndex index = buildIndex(SEED, NUM_VECTORS);
        HnswGraph graph = index.graph();

        int ep = graph.entryPointId();
        assertThat(ep).isGreaterThanOrEqualTo(0);
        assertThat(graph.getNode(ep).maxLevel()).isEqualTo(graph.maxLevel());
    }

    @Test
    @DisplayName("Invariant 4: Determinism with seed - two identical runs produce identical KNN results")
    void testDeterministicWithSeed() {
        HnswIndex indexA = buildIndex(SEED, 100);
        HnswIndex indexB = buildIndex(SEED, 100);

        Random queryRng = new Random(999L);
        for (int q = 0; q < 10; q++) {
            float[] query = randomVector(queryRng, DIMENSION);
            List<SearchResult> resA = indexA.searchKnn(query, 10);
            List<SearchResult> resB = indexB.searchKnn(query, 10);

            assertThat(resA).isEqualTo(resB);
        }
    }

    @Test
    @DisplayName("Invariant 5: Bidirectional connections during construction")
    void testBidirectionalEdges() {
        HnswIndex index = buildIndex(SEED, NUM_VECTORS);
        HnswGraph graph = index.graph();

        int totalDirectedEdges = 0;
        int symmetricEdges = 0;

        for (int i = 0; i < graph.size(); i++) {
            HnswNode node = graph.getNode(i);
            for (int l = 0; l <= node.maxLevel(); l++) {
                for (int neighbor : node.getNeighbors(l)) {
                    totalDirectedEdges++;
                    if (graph.getNode(neighbor).hasNeighbor(l, i)) {
                        symmetricEdges++;
                    }
                }
            }
        }

        // Most edges should be bidirectional. Pruning may occasionally drop one side,
        // but the vast majority of edges are symmetric.
        double symmetryRatio = (double) symmetricEdges / totalDirectedEdges;
        System.out.printf("Edge symmetry ratio: %.2f%% (%d / %d)%n",
                symmetryRatio * 100.0, symmetricEdges, totalDirectedEdges);
        assertThat(symmetryRatio).isGreaterThan(0.70);
    }
}
