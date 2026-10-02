package com.nanovector.server.concurrency;

import com.nanovector.core.hnsw.EpochVisitedSet;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.QuantizedHnswIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.offheap.OffHeapQuantizedHnswIndex;
import com.nanovector.core.search.SearchContextProvider;
import com.nanovector.server.registry.ManagedIndex;
import java.util.List;
import java.util.Objects;

/**
 * Thread-safe search execution coordinator.
 *
 * <p>Ensures that concurrent searches over the same index obtain isolated {@link EpochVisitedSet}
 * context instances from {@link SearchContextProvider}, executing inside {@link
 * ManagedIndex#executeRead(java.util.function.Function)} with shared read lock semantics.
 */
public final class ConcurrentSearchSession {

  private final SearchContextProvider contextProvider;

  public ConcurrentSearchSession(SearchContextProvider contextProvider) {
    this.contextProvider =
        Objects.requireNonNull(contextProvider, "contextProvider must not be null");
  }

  public ConcurrentSearchSession() {
    this(SearchContextProvider.defaultProvider());
  }

  /**
   * Executes a thread-safe k-NN search over a managed index.
   *
   * @param managedIndex index to query
   * @param query query vector
   * @param k number of neighbors
   * @param efSearch exploration width (for graph-based indexes)
   * @return list of nearest neighbors
   */
  public List<SearchResult> search(ManagedIndex managedIndex, float[] query, int k, int efSearch) {
    Objects.requireNonNull(managedIndex, "managedIndex must not be null");
    Objects.requireNonNull(query, "query must not be null");

    int capacityHint = Math.max(1024, managedIndex.metadata().size() + 1);
    EpochVisitedSet visitedSet = contextProvider.acquire(capacityHint);
    try {
      return managedIndex.executeRead(
          idx -> {
            if (idx instanceof HnswIndex hnsw) {
              return hnsw.searchKnn(query, k, efSearch, visitedSet);
            } else if (idx instanceof OffHeapQuantizedHnswIndex offHeap) {
              return offHeap.searchKnn(query, k, efSearch, visitedSet);
            } else if (idx instanceof QuantizedHnswIndex qHnsw) {
              return qHnsw.searchKnn(query, k, efSearch, visitedSet);
            } else {
              return idx.searchKnn(query, k);
            }
          });
    } finally {
      contextProvider.release(visitedSet);
    }
  }

  /**
   * Executes a thread-safe k-NN search using default efSearch or k.
   *
   * @param managedIndex index to query
   * @param query query vector
   * @param k number of neighbors
   * @return list of nearest neighbors
   */
  public List<SearchResult> search(ManagedIndex managedIndex, float[] query, int k) {
    return search(managedIndex, query, k, Math.max(k, 16));
  }

  /** Returns the underlying {@link SearchContextProvider}. */
  public SearchContextProvider contextProvider() {
    return contextProvider;
  }
}
