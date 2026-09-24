package com.nanovector.core.model;

import java.util.Objects;

/**
 * Immutable search result representing a vector match.
 *
 * @param id the external ID of the matched vector
 * @param distance the calculated distance score (smaller is closer/more similar)
 */
public record SearchResult(long id, float distance) implements Comparable<SearchResult> {

  /**
   * Deterministic comparison for ranking: 1. Ascending by {@code distance} (closer matches first).
   * 2. If distances are identical, ascending by {@code id} (deterministic tie-breaking).
   */
  @Override
  public int compareTo(SearchResult other) {
    Objects.requireNonNull(other, "Cannot compare to null SearchResult");
    int cmp = Float.compare(this.distance, other.distance);
    if (cmp != 0) {
      return cmp;
    }
    return Long.compare(this.id, other.id);
  }
}
