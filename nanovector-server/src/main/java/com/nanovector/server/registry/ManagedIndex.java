package com.nanovector.server.registry;

import com.nanovector.core.index.VectorIndex;
import com.nanovector.server.model.IndexMetadata;
import java.util.function.Function;

/**
 * Lifecycle and concurrency manager abstraction for an active vector index.
 *
 * <p>Enforces concurrency control via read/write synchronization and lifecycle state tracking.
 * Guarantees that:
 *
 * <ul>
 *   <li>Concurrent reads (similarity search, metadata inspection) execute without blocking each
 *       other.
 *   <li>Mutations (insert, batch insert) execute with exclusive write semantics.
 *   <li>Closing the index prevents new operations and waits for in-flight operations to drain
 *       before releasing underlying resources (e.g., native off-heap memory segments).
 *   <li>Double closing is strictly idempotent and safe.
 * </ul>
 */
public interface ManagedIndex extends AutoCloseable {

  /** Returns the unique identifier/name of this managed index. */
  String name();

  /** Returns the underlying {@link VectorIndex} instance. */
  VectorIndex index();

  /** Returns the index metadata descriptor. */
  IndexMetadata metadata();

  /**
   * Executes a read-only operation holding the shared read lock.
   *
   * @param action function receiving the underlying index
   * @param <T> result type
   * @return result of the action
   * @throws IllegalStateException if this index is closed or in the process of closing
   */
  <T> T executeRead(Function<VectorIndex, T> action);

  /**
   * Executes an exclusive mutating operation holding the write lock.
   *
   * @param action function receiving the underlying index
   * @param <T> result type
   * @return result of the action
   * @throws IllegalStateException if this index is closed or in the process of closing
   */
  <T> T executeWrite(Function<VectorIndex, T> action);

  /** Returns true if this index has been closed or is shutting down. */
  boolean isClosed();

  /**
   * Atomically closes the index, rejecting any new callers and draining active operations before
   * freeing underlying resources. Idempotent.
   */
  @Override
  void close();
}
