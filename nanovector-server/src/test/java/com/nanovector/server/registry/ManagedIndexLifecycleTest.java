package com.nanovector.server.registry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.offheap.OffHeapQuantizedHnswIndex;
import com.nanovector.server.lifecycle.IndexLifecycleManager;
import com.nanovector.server.model.IndexMetadata;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("ManagedIndexLifecycle Test Suite")
class ManagedIndexLifecycleTest {

  @Test
  @DisplayName("OffHeapQuantizedHnswIndex is closed safely and close is idempotent")
  void offHeapIndexClosedExactlyOnce() {
    HnswConfig config = HnswConfig.defaultConfig();
    OffHeapQuantizedHnswIndex rawIndex =
        new OffHeapQuantizedHnswIndex(16, DistanceMetric.EUCLIDEAN, config);

    ManagedIndex managedIndex =
        IndexLifecycleManager.createManaged(
            "offheap-test",
            rawIndex,
            IndexMetadata.of(
                "offheap-test", "OFFHEAP_QUANTIZED_HNSW", 16, DistanceMetric.EUCLIDEAN, 0));

    assertThat(managedIndex.isClosed()).isFalse();

    // First close
    managedIndex.close();
    assertThat(managedIndex.isClosed()).isTrue();

    // Idempotent second close should not throw
    managedIndex.close();
    managedIndex.close();
    assertThat(managedIndex.isClosed()).isTrue();

    // Subsequent read/write must be rejected immediately
    assertThatThrownBy(() -> managedIndex.executeRead(idx -> idx.size()))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("Index 'offheap-test' is closed");

    assertThatThrownBy(() -> managedIndex.executeWrite(idx -> idx.size()))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("Index 'offheap-test' is closed");
  }

  @Test
  @DisplayName("safelyCreate cleans up native resources on allocation failure without leaking")
  void safelyCreateCleansUpOnFailure() {
    AtomicBoolean closed = new AtomicBoolean(false);

    class LeakingResourceIndex implements com.nanovector.core.index.VectorIndex, AutoCloseable {
      private final OffHeapQuantizedHnswIndex delegate =
          new OffHeapQuantizedHnswIndex(16, DistanceMetric.EUCLIDEAN, HnswConfig.defaultConfig());

      @Override
      public void insert(long id, float[] vector) {
        delegate.insert(id, vector);
      }

      @Override
      public java.util.List<com.nanovector.core.model.SearchResult> searchKnn(
          float[] query, int k) {
        return delegate.searchKnn(query, k);
      }

      @Override
      public int size() {
        return delegate.size();
      }

      @Override
      public int dimension() {
        return delegate.dimension();
      }

      @Override
      public DistanceMetric metric() {
        return delegate.metric();
      }

      @Override
      public com.nanovector.core.storage.VectorDataView vectorData() {
        return delegate.vectorData();
      }

      @Override
      public void close() {
        closed.set(true);
        delegate.close();
      }
    }

    IndexMetadata metadata =
        IndexMetadata.of("fail-test", "OFFHEAP_QUANTIZED_HNSW", 16, DistanceMetric.EUCLIDEAN, 0);

    assertThatThrownBy(
            () ->
                IndexLifecycleManager.safelyCreate(
                    "", // blank name causes DefaultManagedIndex constructor to throw
                    metadata,
                    () -> new LeakingResourceIndex()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Index name must not be null or blank");

    assertThat(closed.get()).isTrue();
  }

  @Test
  @DisplayName("close() blocks and drains active readers before closing native resources")
  void closeDrainsActiveReadersBeforeReleasingResources() throws Exception {
    HnswConfig config = HnswConfig.defaultConfig();
    OffHeapQuantizedHnswIndex rawIndex =
        new OffHeapQuantizedHnswIndex(8, DistanceMetric.EUCLIDEAN, config);

    float[] vec = new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    rawIndex.insert(1L, vec);

    ManagedIndex managedIndex =
        IndexLifecycleManager.createManaged(
            "drain-test",
            rawIndex,
            IndexMetadata.of(
                "drain-test", "OFFHEAP_QUANTIZED_HNSW", 8, DistanceMetric.EUCLIDEAN, 1));

    ExecutorService executor = Executors.newFixedThreadPool(2);
    CountDownLatch readerStarted = new CountDownLatch(1);
    CountDownLatch closeTriggered = new CountDownLatch(1);
    AtomicBoolean readerCompleted = new AtomicBoolean(false);
    AtomicBoolean readerReadSuccess = new AtomicBoolean(false);

    try {
      // Thread 1: Start a long read
      Future<?> readerFuture =
          executor.submit(
              () -> {
                managedIndex.executeRead(
                    idx -> {
                      readerStarted.countDown();
                      try {
                        // Wait until close is triggered by thread 2
                        boolean triggered = closeTriggered.await(2, TimeUnit.SECONDS);
                        assertThat(triggered).isTrue();
                        Thread.sleep(100); // Simulate reading from native memory
                        // Native memory must still be alive and accessible here
                        assertThat(idx.size()).isEqualTo(1);
                        readerReadSuccess.set(true);
                      } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                      }
                      return null;
                    });
                readerCompleted.set(true);
              });

      // Wait until reader has acquired the read lock
      assertThat(readerStarted.await(2, TimeUnit.SECONDS)).isTrue();

      // Thread 2: Trigger close
      Future<?> closeFuture =
          executor.submit(
              () -> {
                closeTriggered.countDown();
                // close() will acquire write lock, which must wait for reader to complete
                managedIndex.close();
                // When close returns, reader MUST have already completed!
                assertThat(readerCompleted.get()).isTrue();
              });

      readerFuture.get(5, TimeUnit.SECONDS);
      closeFuture.get(5, TimeUnit.SECONDS);

      assertThat(readerReadSuccess.get()).isTrue();
      assertThat(managedIndex.isClosed()).isTrue();

      // Subsequent access must fail
      assertThatThrownBy(() -> managedIndex.executeRead(idx -> idx.size()))
          .isInstanceOf(IllegalStateException.class);
    } finally {
      executor.shutdownNow();
    }
  }
}
