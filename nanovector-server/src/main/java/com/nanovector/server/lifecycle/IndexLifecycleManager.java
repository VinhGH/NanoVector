package com.nanovector.server.lifecycle;

import com.nanovector.core.index.VectorIndex;
import com.nanovector.server.model.IndexMetadata;
import com.nanovector.server.registry.DefaultManagedIndex;
import com.nanovector.server.registry.ManagedIndex;
import java.util.Objects;

/**
 * Lifecycle orchestrator for safely constructing, managing, and disposing vector indexes.
 *
 * <p>Guarantees that index allocation failures do not leak off-heap native memory segments or file
 * descriptors: if construction or initialization fails at any point after allocating native
 * segments, {@link #safelyCreate(String, IndexMetadata, ThrowingSupplier)} ensures the allocated
 * resources are immediately closed before propagating the failure.
 */
public final class IndexLifecycleManager {

  @FunctionalInterface
  public interface ThrowingSupplier<T> {
    T get() throws Exception;
  }

  private IndexLifecycleManager() {}

  /**
   * Wraps an existing {@link VectorIndex} and {@link IndexMetadata} in a {@link
   * DefaultManagedIndex}.
   *
   * @param name index identifier
   * @param index vector index instance
   * @param metadata metadata descriptor
   * @return managed index instance
   */
  public static ManagedIndex createManaged(String name, VectorIndex index, IndexMetadata metadata) {
    return new DefaultManagedIndex(name, index, metadata);
  }

  /**
   * Executes index allocation and wrapping inside a safe lifecycle guard.
   *
   * <p>If {@code supplier.get()} succeeds but subsequent initialization or wrapping throws an
   * exception, or if a supplier produces a partially configured {@link AutoCloseable} index before
   * failing, any created native resources are strictly closed.
   *
   * @param name index identifier
   * @param metadata metadata descriptor
   * @param supplier factory producing the vector index
   * @param <T> concrete vector index type
   * @return managed index instance
   * @throws Exception if construction fails
   */
  public static <T extends VectorIndex> ManagedIndex safelyCreate(
      String name, IndexMetadata metadata, ThrowingSupplier<T> supplier) throws Exception {
    Objects.requireNonNull(name, "name must not be null");
    Objects.requireNonNull(metadata, "metadata must not be null");
    Objects.requireNonNull(supplier, "supplier must not be null");

    T index = null;
    try {
      index = supplier.get();
      return new DefaultManagedIndex(name, index, metadata);
    } catch (Throwable t) {
      if (index instanceof AutoCloseable autoCloseable) {
        try {
          autoCloseable.close();
        } catch (Exception closeEx) {
          t.addSuppressed(closeEx);
        }
      }
      if (t instanceof Exception e) {
        throw e;
      }
      throw new RuntimeException("Unexpected error during index creation: " + t.getMessage(), t);
    }
  }
}
