package com.nanovector.core.search;

import com.nanovector.core.hnsw.EpochVisitedSet;
import java.util.Objects;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * High-performance, thread-safe object pool and provider for {@link EpochVisitedSet} instances.
 *
 * <p>During concurrent graph searches, sharing a single visited set causes race conditions, epoch
 * collisions, and incorrect nearest-neighbor candidates. Allocating a new visited set on every
 * query introduces unnecessary heap garbage collection churn.
 *
 * <p>{@code SearchContextProvider} resolves this by pooling recyclable {@link EpochVisitedSet}
 * instances using a non-blocking {@link ConcurrentLinkedQueue}. When a thread acquires a set, it is
 * guaranteed isolated ownership for the duration of the query. Upon search completion, the set is
 * returned to the pool via {@link #release(EpochVisitedSet)} or auto-closed using {@link
 * #acquireLease(int)}.
 */
public final class SearchContextProvider {

  private static final SearchContextProvider DEFAULT_INSTANCE = new SearchContextProvider();

  private static final int DEFAULT_INITIAL_CAPACITY = 1024;
  private static final int DEFAULT_MAX_POOL_SIZE = 128;

  private final Queue<EpochVisitedSet> pool;
  private final int defaultInitialCapacity;
  private final int maxPoolSize;

  public SearchContextProvider() {
    this(DEFAULT_INITIAL_CAPACITY, DEFAULT_MAX_POOL_SIZE);
  }

  public SearchContextProvider(int defaultInitialCapacity, int maxPoolSize) {
    if (defaultInitialCapacity <= 0) {
      throw new IllegalArgumentException(
          "defaultInitialCapacity must be positive: " + defaultInitialCapacity);
    }
    if (maxPoolSize <= 0) {
      throw new IllegalArgumentException("maxPoolSize must be positive: " + maxPoolSize);
    }
    this.pool = new ConcurrentLinkedQueue<>();
    this.defaultInitialCapacity = defaultInitialCapacity;
    this.maxPoolSize = maxPoolSize;
  }

  /** Returns the global shared default search context provider instance. */
  public static SearchContextProvider defaultProvider() {
    return DEFAULT_INSTANCE;
  }

  /**
   * Acquires an {@link EpochVisitedSet} from the pool or allocates a new one if the pool is empty.
   * Ensures that the returned set has at least {@code minCapacity}.
   *
   * @param minCapacity minimum required capacity to accommodate index nodes
   * @return an isolated {@link EpochVisitedSet} instance
   */
  public EpochVisitedSet acquire(int minCapacity) {
    int cap = Math.max(minCapacity, defaultInitialCapacity);
    EpochVisitedSet set = pool.poll();
    if (set == null) {
      return new EpochVisitedSet(cap);
    }
    set.ensureCapacity(cap);
    return set;
  }

  /**
   * Acquires an {@link EpochVisitedSet} with default capacity.
   *
   * @return an isolated {@link EpochVisitedSet} instance
   */
  public EpochVisitedSet acquire() {
    return acquire(defaultInitialCapacity);
  }

  /**
   * Acquires a scoped {@link ContextLease} wrapping an acquired {@link EpochVisitedSet}.
   *
   * @param minCapacity minimum required capacity
   * @return an {@link AutoCloseable} lease that returns the set to this provider on close
   */
  public ContextLease acquireLease(int minCapacity) {
    return new ContextLease(acquire(minCapacity), this);
  }

  /**
   * Acquires a scoped {@link ContextLease} with default capacity.
   *
   * @return an {@link AutoCloseable} lease that returns the set to this provider on close
   */
  public ContextLease acquireLease() {
    return acquireLease(defaultInitialCapacity);
  }

  /**
   * Releases an {@link EpochVisitedSet} back to the pool for reuse by subsequent queries.
   *
   * @param set visited set to recycle
   */
  public void release(EpochVisitedSet set) {
    if (set != null && pool.size() < maxPoolSize) {
      pool.offer(set);
    }
  }

  /** Returns the number of currently idle pooled instances. */
  public int idleCount() {
    return pool.size();
  }

  /** Clears all pooled instances. */
  public void clear() {
    pool.clear();
  }

  /**
   * Scoped handle for automatic release of an acquired {@link EpochVisitedSet} via
   * try-with-resources.
   */
  public record ContextLease(EpochVisitedSet visitedSet, SearchContextProvider provider)
      implements AutoCloseable {

    public ContextLease {
      Objects.requireNonNull(visitedSet, "visitedSet must not be null");
      Objects.requireNonNull(provider, "provider must not be null");
    }

    @Override
    public void close() {
      provider.release(visitedSet);
    }
  }
}
