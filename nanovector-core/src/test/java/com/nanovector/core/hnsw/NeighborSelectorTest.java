package com.nanovector.core.hnsw;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class NeighborSelectorTest {

  @Test
  @DisplayName(
      "NeighborSelector: Algorithm 4 prefers angular diversity over collinear occluded nodes")
  void testAlgorithm4HeuristicPrefersAngularDiversity() {
    // Target U is at (0, 0)
    // Candidate 1 (A) at (1, 0) -> dist to target = 1.0
    // Candidate 2 (B) at (2, 0) -> dist to target = 2.0 (collinear behind A)
    // Candidate 3 (C) at (0, 1.2) -> dist to target = 1.2 (orthogonal to A)
    // Candidate 4 (D) at (0, 2.5) -> dist to target = 2.5 (collinear behind C)
    float[][] coords = {
      {0.0f, 0.0f}, // 0: Target U
      {1.0f, 0.0f}, // 1: A
      {2.0f, 0.0f}, // 2: B
      {0.0f, 1.2f}, // 3: C
      {0.0f, 2.5f} // 4: D
    };

    NeighborSelector.NodeDistanceEvaluator evaluator =
        (a, b) -> {
          float dx = coords[a][0] - coords[b][0];
          float dy = coords[a][1] - coords[b][1];
          return (float) Math.sqrt(dx * dx + dy * dy);
        };

    List<NeighborSelector.Candidate> candidates = new ArrayList<>();
    candidates.add(new NeighborSelector.Candidate(1, evaluator.distance(0, 1))); // 1.0
    candidates.add(new NeighborSelector.Candidate(2, evaluator.distance(0, 2))); // 2.0
    candidates.add(new NeighborSelector.Candidate(3, evaluator.distance(0, 3))); // 1.2
    candidates.add(new NeighborSelector.Candidate(4, evaluator.distance(0, 4))); // 2.5

    // Request maxDegree = 2
    // Algorithm 4 should select A (dist 1.0), then C (dist 1.2), rejecting B because B is closer to
    // A (dist 1.0 <= 2.0)
    int[] selected = NeighborSelector.selectNeighbors(candidates, 2, evaluator);

    assertThat(selected).hasSize(2);
    assertThat(selected[0]).isEqualTo(1); // Node A
    assertThat(selected[1]).isEqualTo(3); // Node C (angular diversity chosen over B)
  }

  @Test
  @DisplayName(
      "NeighborSelector: Fallback fills remaining capacity when heuristic prunes too aggressively")
  void testFallbackMechanismFillsCapacity() {
    // All candidates are collinear on the x-axis:
    // Target U at (0, 0)
    // Node 1 at (1, 0) -> dist 1.0
    // Node 2 at (2, 0) -> dist 2.0 (dist to 1 is 1.0 <= 2.0)
    // Node 3 at (3, 0) -> dist 3.0 (dist to 2 is 1.0 <= 3.0)
    float[][] coords = {
      {0.0f, 0.0f}, // 0: Target U
      {1.0f, 0.0f}, // 1
      {2.0f, 0.0f}, // 2
      {3.0f, 0.0f} // 3
    };

    NeighborSelector.NodeDistanceEvaluator evaluator =
        (a, b) -> {
          float dx = coords[a][0] - coords[b][0];
          float dy = coords[a][1] - coords[b][1];
          return (float) Math.abs(dx);
        };

    List<NeighborSelector.Candidate> candidates = new ArrayList<>();
    candidates.add(new NeighborSelector.Candidate(1, evaluator.distance(0, 1)));
    candidates.add(new NeighborSelector.Candidate(2, evaluator.distance(0, 2)));
    candidates.add(new NeighborSelector.Candidate(3, evaluator.distance(0, 3)));

    // Request maxDegree = 2
    // Pure heuristic would only select Node 1 because Nodes 2 & 3 are occluded.
    // NanoVector fallback fills the second spot with Node 2 (nearest discarded).
    int[] selected = NeighborSelector.selectNeighbors(candidates, 2, evaluator);

    assertThat(selected).hasSize(2);
    assertThat(selected[0]).isEqualTo(1);
    assertThat(selected[1]).isEqualTo(2); // Added via fallback
  }

  @Test
  @DisplayName(
      "NeighborSelector: Deterministic tie-breaking on identical distance orders by ID ascending")
  void testDeterministicTieBreaking() {
    List<NeighborSelector.Candidate> candidates = new ArrayList<>();
    candidates.add(new NeighborSelector.Candidate(50, 1.5f));
    candidates.add(new NeighborSelector.Candidate(20, 1.5f));
    candidates.add(new NeighborSelector.Candidate(80, 1.5f));

    // Infinite distance between candidate nodes so none occlude each other
    NeighborSelector.NodeDistanceEvaluator evaluator = (a, b) -> 100.0f;

    int[] selected = NeighborSelector.selectNeighbors(candidates, 2, evaluator);

    assertThat(selected).hasSize(2);
    assertThat(selected[0]).isEqualTo(20);
    assertThat(selected[1]).isEqualTo(50);
  }

  @Test
  @DisplayName("NeighborSelector: Edge cases and validation")
  void testEdgeCases() {
    NeighborSelector.NodeDistanceEvaluator evaluator = (a, b) -> 1.0f;

    // Empty candidates
    int[] emptyResult = NeighborSelector.selectNeighbors(new ArrayList<>(), 5, evaluator);
    assertThat(emptyResult).isEmpty();

    // Fewer candidates than maxDegree
    List<NeighborSelector.Candidate> smallList = new ArrayList<>();
    smallList.add(new NeighborSelector.Candidate(10, 1.0f));
    int[] smallResult = NeighborSelector.selectNeighbors(smallList, 5, evaluator);
    assertThat(smallResult).containsExactly(10);

    // Invalid maxDegree
    assertThatThrownBy(() -> NeighborSelector.selectNeighbors(smallList, 0, evaluator))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
