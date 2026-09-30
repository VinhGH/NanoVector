package com.nanovector.core.offheap;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.core.hnsw.EpochVisitedSet;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.hnsw.LevelGenerator;
import com.nanovector.core.hnsw.NeighborSelector;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class OffHeapVsOnHeapGraphEquivalenceTest {

  @Test
  @DisplayName(
      "Should maintain 100% behavioral equivalence between On-Heap and Off-Heap HNSW graphs")
  void shouldMaintainStrictBehavioralEquivalence() {
    int dimension = 32;
    int numNodes = 250;
    long seed = 42L;

    HnswConfig config = new HnswConfig(16, 32, 100, 50, 1.0 / Math.log(16), seed);

    Random random = new Random(seed);
    float[][] vectors = new float[numNodes][dimension];
    for (int i = 0; i < numNodes; i++) {
      for (int d = 0; d < dimension; d++) {
        vectors[i][d] = random.nextFloat() * 2.0f - 1.0f;
      }
    }

    HnswGraph onHeapGraph = new HnswGraph(config);
    try (OffHeapHnswGraph offHeapGraph = new OffHeapHnswGraph(config, numNodes)) {
      LevelGenerator levelGen = new LevelGenerator(config);
      EpochVisitedSet onHeapVisited = new EpochVisitedSet(numNodes);
      EpochVisitedSet offHeapVisited = new EpochVisitedSet(numNodes);

      NeighborSelector.NodeDistanceEvaluator evaluator =
          (u, v) -> squaredL2(vectors[u], vectors[v]);

      // Step 1: Insert all nodes into both graphs with identical logic
      for (int i = 0; i < numNodes; i++) {
        int internalId = i;
        int nodeLevel = levelGen.nextLevel();
        float[] query = vectors[internalId];

        HnswGraph.DistanceToQuery distToQuery = node -> squaredL2(query, vectors[node]);

        // Insert into On-Heap
        insertIntoOnHeap(
            onHeapGraph, internalId, nodeLevel, distToQuery, evaluator, onHeapVisited, config);

        // Insert into Off-Heap
        insertIntoOffHeap(
            offHeapGraph, internalId, nodeLevel, distToQuery, evaluator, offHeapVisited, config);
      }

      // Step 2: Verify structural parity
      assertThat(offHeapGraph.size()).isEqualTo(onHeapGraph.size());
      assertThat(offHeapGraph.entryPointId()).isEqualTo(onHeapGraph.entryPointId());
      assertThat(offHeapGraph.maxLevel()).isEqualTo(onHeapGraph.maxLevel());

      for (int i = 0; i < numNodes; i++) {
        HnswNode onHeapNode = onHeapGraph.getNode(i);
        int maxL = onHeapNode.maxLevel();
        assertThat(offHeapGraph.layout().maxLevel(i)).isEqualTo(maxL);

        for (int l = 0; l <= maxL; l++) {
          int[] onHeapNeighbors = onHeapNode.getNeighbors(l);
          int[] offHeapNeighbors = offHeapGraph.getNeighbors(i, l);

          assertThat(offHeapNeighbors)
              .as("Neighbors mismatch at node %d, layer %d", i, l)
              .containsExactly(onHeapNeighbors);

          assertThat(offHeapGraph.degree(i, l)).isEqualTo(onHeapNode.degree(l));
        }
      }

      // Step 3: Verify bidirectional edge symmetry invariant on both graphs
      for (int i = 0; i < numNodes; i++) {
        int maxL = offHeapGraph.layout().maxLevel(i);
        for (int l = 0; l <= maxL; l++) {
          for (int neighborId : offHeapGraph.getNeighbors(i, l)) {
            assertThat(offHeapGraph.hasNeighbor(neighborId, l, i))
                .as("Off-heap symmetry broken between %d and %d at layer %d", i, neighborId, l)
                .isTrue();
            assertThat(onHeapGraph.getNode(neighborId).hasNeighbor(l, i))
                .as("On-heap symmetry broken between %d and %d at layer %d", i, neighborId, l)
                .isTrue();
          }
        }
      }

      // Step 4: Verify search traversal parity (greedy descent & searchLayer) across 50 queries
      Random queryRand = new Random(12345L);
      for (int q = 0; q < 50; q++) {
        float[] queryVec = new float[dimension];
        for (int d = 0; d < dimension; d++) {
          queryVec[d] = queryRand.nextFloat() * 2.0f - 1.0f;
        }

        HnswGraph.DistanceToQuery distFunc = node -> squaredL2(queryVec, vectors[node]);

        // Verify greedy descent parity across upper layers
        int currentBestOnHeap = onHeapGraph.entryPointId();
        int currentBestOffHeap = offHeapGraph.entryPointId();
        assertThat(currentBestOffHeap).isEqualTo(currentBestOnHeap);

        for (int l = onHeapGraph.maxLevel(); l >= 1; l--) {
          currentBestOnHeap = onHeapGraph.greedyClosest(distFunc, currentBestOnHeap, l);
          currentBestOffHeap = offHeapGraph.greedyClosest(distFunc, currentBestOffHeap, l);
          assertThat(currentBestOffHeap)
              .as("Greedy routing mismatch at layer %d for query %d", l, q)
              .isEqualTo(currentBestOnHeap);
        }

        // Verify searchLayer parity with varying efSearch
        for (int ef : new int[] {10, 30, 64}) {
          onHeapVisited.nextEpoch();
          offHeapVisited.nextEpoch();

          List<NeighborSelector.Candidate> onHeapResults =
              onHeapGraph.searchLayer(
                  distFunc, new int[] {currentBestOnHeap}, ef, 0, onHeapVisited);

          List<NeighborSelector.Candidate> offHeapResults =
              offHeapGraph.searchLayer(
                  distFunc, new int[] {currentBestOffHeap}, ef, 0, offHeapVisited);

          assertThat(offHeapResults).hasSameSizeAs(onHeapResults);
          for (int k = 0; k < onHeapResults.size(); k++) {
            assertThat(offHeapResults.get(k).id())
                .as("Search candidate ID mismatch at rank %d (ef=%d)", k, ef)
                .isEqualTo(onHeapResults.get(k).id());
            assertThat(offHeapResults.get(k).distance())
                .as("Search candidate distance mismatch at rank %d (ef=%d)", k, ef)
                .isEqualTo(onHeapResults.get(k).distance());
          }
        }
      }
    }
  }

  private static void insertIntoOnHeap(
      HnswGraph graph,
      int internalId,
      int nodeLevel,
      HnswGraph.DistanceToQuery distToQuery,
      NeighborSelector.NodeDistanceEvaluator evaluator,
      EpochVisitedSet visitedSet,
      HnswConfig config) {
    HnswNode node = new HnswNode(internalId, nodeLevel);
    graph.addNode(node);

    if (graph.size() == 1) {
      return;
    }

    int currentBest = graph.entryPointId();
    int graphMaxLevel = graph.maxLevel();

    for (int l = graphMaxLevel; l > nodeLevel; l--) {
      if (graph.getNode(currentBest).maxLevel() >= l) {
        currentBest = graph.greedyClosest(distToQuery, currentBest, l);
      }
    }

    int insertionTopLayer = Math.min(graphMaxLevel, nodeLevel);
    for (int l = insertionTopLayer; l >= 0; l--) {
      visitedSet.nextEpoch();
      List<NeighborSelector.Candidate> candidates =
          graph.searchLayer(
              distToQuery, new int[] {currentBest}, config.efConstruction(), l, visitedSet);

      int maxDegree = graph.maxDegreeForLayer(l);
      int[] selectedNeighbors =
          NeighborSelector.selectNeighbors(new ArrayList<>(candidates), maxDegree, evaluator);

      for (int neighborId : selectedNeighbors) {
        graph.connect(internalId, neighborId, l, evaluator);
      }

      if (!candidates.isEmpty()) {
        currentBest = candidates.get(0).id();
      }
    }

    if (nodeLevel > graphMaxLevel) {
      graph.setEntryPoint(internalId, nodeLevel);
    }
  }

  private static void insertIntoOffHeap(
      OffHeapHnswGraph graph,
      int internalId,
      int nodeLevel,
      HnswGraph.DistanceToQuery distToQuery,
      NeighborSelector.NodeDistanceEvaluator evaluator,
      EpochVisitedSet visitedSet,
      HnswConfig config) {
    graph.addNode(internalId, nodeLevel);

    if (graph.size() == 1) {
      return;
    }

    int currentBest = graph.entryPointId();
    int graphMaxLevel = graph.maxLevel();

    for (int l = graphMaxLevel; l > nodeLevel; l--) {
      if (graph.layout().maxLevel(currentBest) >= l) {
        currentBest = graph.greedyClosest(distToQuery, currentBest, l);
      }
    }

    int insertionTopLayer = Math.min(graphMaxLevel, nodeLevel);
    for (int l = insertionTopLayer; l >= 0; l--) {
      visitedSet.nextEpoch();
      List<NeighborSelector.Candidate> candidates =
          graph.searchLayer(
              distToQuery, new int[] {currentBest}, config.efConstruction(), l, visitedSet);

      int maxDegree = graph.maxDegreeForLayer(l);
      int[] selectedNeighbors =
          NeighborSelector.selectNeighbors(new ArrayList<>(candidates), maxDegree, evaluator);

      for (int neighborId : selectedNeighbors) {
        graph.connect(internalId, neighborId, l, evaluator);
      }

      if (!candidates.isEmpty()) {
        currentBest = candidates.get(0).id();
      }
    }

    if (nodeLevel > graphMaxLevel) {
      graph.setEntryPoint(internalId, nodeLevel);
    }
  }

  private static float squaredL2(float[] a, float[] b) {
    float sum = 0.0f;
    for (int i = 0; i < a.length; i++) {
      float diff = a[i] - b[i];
      sum += diff * diff;
    }
    return sum;
  }
}
