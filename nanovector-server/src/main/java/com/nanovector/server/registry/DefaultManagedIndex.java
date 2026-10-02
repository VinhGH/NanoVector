package com.nanovector.server.registry;

import com.nanovector.core.index.VectorIndex;
import com.nanovector.server.model.IndexMetadata;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Function;

/**
 * Standard implementation of {@link ManagedIndex} employing fair read-write locking and atomic
 * lifecycle guarantees.
 *
 * <p>Protects native memory segments from use-after-free and concurrent corruption:
 *
 * <ol>
 *   <li>{@link #close()} atomically toggles an {@link AtomicBoolean} to immediately reject any new
 *       incoming readers or writers.
 *   <li>{@link #close()} subsequently acquires the exclusive write lock, forcing it to block until
 *       all currently active search or read operations release their read locks.
 *   <li>Only when all active readers have cleanly completed is the underlying {@link AutoCloseable}
 *       resource closed.
 * </ol>
 */
public final class DefaultManagedIndex implements ManagedIndex {

  private final String name;
  private final VectorIndex index;
  private final IndexMetadata metadata;
  private final ReentrantReadWriteLock rwLock;
  private final AtomicBoolean closed;

  public DefaultManagedIndex(String name, VectorIndex index, IndexMetadata metadata) {
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("Index name must not be null or blank");
    }
    this.name = name;
    this.index = Objects.requireNonNull(index, "VectorIndex must not be null");
    this.metadata = Objects.requireNonNull(metadata, "IndexMetadata must not be null");
    this.rwLock = new ReentrantReadWriteLock(true); // Fair locking policy
    this.closed = new AtomicBoolean(false);
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public VectorIndex index() {
    return index;
  }

  @Override
  public IndexMetadata metadata() {
    return metadata;
  }

  @Override
  public <T> T executeRead(Function<VectorIndex, T> action) {
    Objects.requireNonNull(action, "action must not be null");
    if (closed.get()) {
      throw new IllegalStateException("Index '" + name + "' is closed");
    }

    rwLock.readLock().lock();
    try {
      if (closed.get()) {
        throw new IllegalStateException("Index '" + name + "' is closed");
      }
      return action.apply(index);
    } finally {
      rwLock.readLock().unlock();
    }
  }

  @Override
  public <T> T executeWrite(Function<VectorIndex, T> action) {
    Objects.requireNonNull(action, "action must not be null");
    if (closed.get()) {
      throw new IllegalStateException("Index '" + name + "' is closed");
    }

    rwLock.writeLock().lock();
    try {
      if (closed.get()) {
        throw new IllegalStateException("Index '" + name + "' is closed");
      }
      T result = action.apply(index);
      metadata.updateSize(index.size());
      metadata.updateLastModified();
      return result;
    } finally {
      rwLock.writeLock().unlock();
    }
  }

  @Override
  public boolean isClosed() {
    return closed.get();
  }

  @Override
  public void close() {
    if (!closed.compareAndSet(false, true)) {
      return; // Already closed or currently closing
    }

    // Acquire write lock to drain all active readers and writers before closing resources
    rwLock.writeLock().lock();
    try {
      if (index instanceof AutoCloseable autoCloseable) {
        try {
          autoCloseable.close();
        } catch (Exception e) {
          throw new RuntimeException(
              "Failed to close underlying resources for index '" + name + "': " + e.getMessage(),
              e);
        }
      }
    } finally {
      rwLock.writeLock().unlock();
    }
  }
}
