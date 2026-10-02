package com.nanovector.core.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.hnsw.EpochVisitedSet;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("SearchContextProvider Test Suite")
class SearchContextProviderTest {

  @Test
  @DisplayName("Acquires and releases EpochVisitedSet correctly with pooling")
  void acquiresAndReleasesWithPooling() {
    SearchContextProvider provider = new SearchContextProvider(512, 10);
    assertThat(provider.idleCount()).isEqualTo(0);

    EpochVisitedSet set1 = provider.acquire(1000);
    assertThat(set1.capacity()).isGreaterThanOrEqualTo(1000);
    assertThat(provider.idleCount()).isEqualTo(0);

    provider.release(set1);
    assertThat(provider.idleCount()).isEqualTo(1);

    EpochVisitedSet recycled = provider.acquire(500);
    assertThat(recycled).isSameAs(set1);
    assertThat(provider.idleCount()).isEqualTo(0);
  }

  @Test
  @DisplayName("AutoCloseable ContextLease recycles instance upon exit")
  void contextLeaseRecyclesOnClose() {
    SearchContextProvider provider = new SearchContextProvider(256, 10);

    EpochVisitedSet captured;
    try (var lease = provider.acquireLease(512)) {
      captured = lease.visitedSet();
      assertThat(captured).isNotNull();
      assertThat(captured.capacity()).isGreaterThanOrEqualTo(512);
      assertThat(provider.idleCount()).isEqualTo(0);
    }

    assertThat(provider.idleCount()).isEqualTo(1);
    EpochVisitedSet recycled = provider.acquire();
    assertThat(recycled).isSameAs(captured);
  }

  @Test
  @DisplayName("Respects maximum pool capacity")
  void respectsMaxPoolSize() {
    SearchContextProvider provider = new SearchContextProvider(128, 2);

    EpochVisitedSet s1 = provider.acquire();
    EpochVisitedSet s2 = provider.acquire();
    EpochVisitedSet s3 = provider.acquire();

    provider.release(s1);
    provider.release(s2);
    provider.release(s3); // Should be discarded because max size is 2

    assertThat(provider.idleCount()).isEqualTo(2);
    provider.clear();
    assertThat(provider.idleCount()).isEqualTo(0);
  }

  @Test
  @DisplayName("Multi-threaded acquisition returns distinct isolated instances concurrently")
  void multiThreadedAcquisitionReturnsDistinctInstances() throws Exception {
    SearchContextProvider provider = new SearchContextProvider(128, 64);
    int threadCount = 16;
    ExecutorService executor = Executors.newFixedThreadPool(threadCount);

    try {
      List<Callable<EpochVisitedSet>> tasks = new ArrayList<>();
      for (int i = 0; i < threadCount; i++) {
        tasks.add(
            () -> {
              EpochVisitedSet set = provider.acquire(256);
              // Mark node and simulate small workload
              set.markVisited(42);
              Thread.sleep(10);
              assertThat(set.isVisited(42)).isTrue();
              return set;
            });
      }

      List<Future<EpochVisitedSet>> futures = executor.invokeAll(tasks);
      List<EpochVisitedSet> acquiredSets = new ArrayList<>();
      for (Future<EpochVisitedSet> f : futures) {
        acquiredSets.add(f.get());
      }

      // All acquired instances held concurrently must be distinct references
      long distinctCount = acquiredSets.stream().distinct().count();
      assertThat(distinctCount).isEqualTo(threadCount);

      // Release all back
      acquiredSets.forEach(provider::release);
      assertThat(provider.idleCount()).isEqualTo(threadCount);
    } finally {
      executor.shutdownNow();
    }
  }

  @Test
  @DisplayName("Rejects invalid constructor arguments")
  void rejectsInvalidConstructorArguments() {
    assertThatThrownBy(() -> new SearchContextProvider(0, 10))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new SearchContextProvider(100, 0))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
