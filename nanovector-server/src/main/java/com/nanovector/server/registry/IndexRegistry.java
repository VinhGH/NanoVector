package com.nanovector.server.registry;

import jakarta.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Thread-safe central registry for active {@link ManagedIndex} instances in NanoVector Server.
 *
 * <p>Coordinates index registration, namespace isolation, thread-safe lookup, deletion with
 * resource deallocation, and safe graceful application shutdown.
 */
public final class IndexRegistry implements AutoCloseable {

  private final ConcurrentMap<String, ManagedIndex> registry = new ConcurrentHashMap<>();

  /**
   * Registers a new {@link ManagedIndex}.
   *
   * @param managedIndex index to register
   * @throws IllegalArgumentException if name is already registered or invalid
   */
  public void register(ManagedIndex managedIndex) {
    Objects.requireNonNull(managedIndex, "managedIndex must not be null");
    String name = managedIndex.name();
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("Index name must not be null or blank");
    }

    ManagedIndex existing = registry.putIfAbsent(name, managedIndex);
    if (existing != null) {
      throw new IllegalArgumentException("Index already exists: " + name);
    }
  }

  /**
   * Retrieves a managed index by name.
   *
   * @param name index identifier
   * @return optional containing the index if found
   */
  public Optional<ManagedIndex> get(String name) {
    if (name == null) {
      return Optional.empty();
    }
    return Optional.ofNullable(registry.get(name));
  }

  /**
   * Retrieves a managed index by name or throws {@link NoSuchElementException} if not present.
   *
   * @param name index identifier
   * @return the managed index
   * @throws NoSuchElementException if index not found
   */
  public ManagedIndex getRequired(String name) {
    return get(name).orElseThrow(() -> new NoSuchElementException("Index not found: " + name));
  }

  /**
   * Deletes and closes an index by name without impacting other registered indices.
   *
   * @param name index identifier to remove
   * @return true if index existed and was closed, false if not found
   */
  public boolean delete(String name) {
    if (name == null) {
      return false;
    }
    ManagedIndex removed = registry.remove(name);
    if (removed != null) {
      removed.close();
      return true;
    }
    return false;
  }

  /** Checks whether an index exists under the given name. */
  public boolean contains(String name) {
    return name != null && registry.containsKey(name);
  }

  /** Returns an unmodifiable snapshot list of all currently registered managed indices. */
  public List<ManagedIndex> list() {
    return Collections.unmodifiableList(new ArrayList<>(registry.values()));
  }

  /** Returns the total number of registered indices. */
  public int size() {
    return registry.size();
  }

  /**
   * Graceful shutdown hook. Closes all registered indices, ensuring off-heap native memory segments
   * and background structures are properly freed.
   */
  @Override
  @PreDestroy
  public void close() {
    List<ManagedIndex> activeIndices = new ArrayList<>(registry.values());
    registry.clear();
    for (ManagedIndex idx : activeIndices) {
      try {
        idx.close();
      } catch (Exception ignored) {
        // Suppress individual closure failures during mass shutdown to ensure all indices are
        // closed
      }
    }
  }
}
