package com.nanovector.core.hnsw;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Neighbor selection heuristic based on Algorithm 4 from Malkov &amp; Yashunin (2018),
 * with a deterministic fallback to prevent unnecessarily sparse local neighborhoods.
 * <p>
 * Standard nearest-neighbor selection (picking the closest $M$ candidates) often causes
 * clustered edges in the same direction, reducing angular diversity.
 * Algorithm 4 ensures diverse edge directions by selecting a candidate only if it is
 * closer to the target than to any neighbor already selected.
 * <p>
 * <b>NanoVector Fallback:</b>
 * If the strict heuristic selects fewer than {@code maxDegree} neighbors and unselected
 * candidates remain, the nearest remaining candidates are added up to {@code maxDegree}.
 */
public final class NeighborSelector {

    @FunctionalInterface
    public interface NodeDistanceEvaluator {
        /**
         * Computes the distance between two internal nodes.
         *
         * @param nodeA internal ID of first node
         * @param nodeB internal ID of second node
         * @return distance between nodeA and nodeB
         */
        float distance(int nodeA, int nodeB);
    }

    /**
     * Candidate representation holding internal node ID and its distance to the target.
     */
    public record Candidate(int id, float distance) implements Comparable<Candidate> {
        @Override
        public int compareTo(Candidate other) {
            Objects.requireNonNull(other, "Cannot compare with null candidate");
            int cmp = Float.compare(this.distance, other.distance);
            if (cmp != 0) {
                return cmp;
            }
            return Integer.compare(this.id, other.id);
        }
    }

    private NeighborSelector() {
        // Utility class
    }

    /**
     * Selects up to {@code maxDegree} neighbors using Algorithm 4 heuristic with fallback.
     *
     * @param candidates        candidate list (will be sorted ascending by distance to target)
     * @param maxDegree         maximum number of neighbors to select (M or M0)
     * @param distanceEvaluator evaluator to compute distances between candidate nodes
     * @return array of selected internal node IDs
     */
    public static int[] selectNeighbors(
            List<Candidate> candidates,
            int maxDegree,
            NodeDistanceEvaluator distanceEvaluator
    ) {
        Objects.requireNonNull(candidates, "Candidates list must not be null");
        Objects.requireNonNull(distanceEvaluator, "DistanceEvaluator must not be null");
        if (maxDegree <= 0) {
            throw new IllegalArgumentException("maxDegree must be positive, but got: " + maxDegree);
        }
        if (candidates.isEmpty()) {
            return new int[0];
        }

        // Sort candidates ascending by distance to target, then by ID for deterministic order
        Collections.sort(candidates);

        int[] selected = new int[maxDegree];
        int selectedCount = 0;
        List<Candidate> discarded = new ArrayList<>();

        // Phase 1: Algorithm 4 Heuristic selection
        for (Candidate candidate : candidates) {
            if (selectedCount >= maxDegree) {
                break;
            }

            int e = candidate.id();
            float distToTarget = candidate.distance();
            boolean isCloserToTargetThanAnyNeighbor = true;

            for (int i = 0; i < selectedCount; i++) {
                int r = selected[i];
                if (e == r) {
                    isCloserToTargetThanAnyNeighbor = false;
                    break;
                }
                float distToSelected = distanceEvaluator.distance(e, r);
                if (distToSelected <= distToTarget) {
                    // e is closer to r than to target (e is occluded by r)
                    isCloserToTargetThanAnyNeighbor = false;
                    break;
                }
            }

            if (isCloserToTargetThanAnyNeighbor) {
                selected[selectedCount++] = e;
            } else {
                discarded.add(candidate);
            }
        }

        // Phase 2: NanoVector Fallback
        // If heuristic selected fewer than maxDegree neighbors, fill from nearest discarded candidates
        if (selectedCount < maxDegree && !discarded.isEmpty()) {
            for (Candidate fallback : discarded) {
                if (selectedCount >= maxDegree) {
                    break;
                }
                // Avoid self-loop or duplicate if present
                boolean alreadySelected = false;
                for (int i = 0; i < selectedCount; i++) {
                    if (selected[i] == fallback.id()) {
                        alreadySelected = true;
                        break;
                    }
                }
                if (!alreadySelected) {
                    selected[selectedCount++] = fallback.id();
                }
            }
        }

        return selectedCount == maxDegree ? selected : Arrays.copyOf(selected, selectedCount);
    }
}
