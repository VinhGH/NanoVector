package com.nanovector.server.concurrency;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.offheap.OffHeapQuantizedHnswIndex;
import com.nanovector.server.lifecycle.IndexLifecycleManager;
import com.nanovector.server.model.IndexMetadata;
import com.nanovector.server.registry.ManagedIndex;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("ConcurrentSearchSession Concurrency & Visited Isolation Test Suite")
class ConcurrentSearchSessionTest {

  private static final int DIMENSION = 16;
  private static final int NUM_VECTORS = 250;
  private static final int NUM_THREADS = 20;
  private static final int K = 5;
  private static final int EF_SEARCH = 32;

  @Test
  @DisplayName(
      "20 concurrent threads querying the SAME HNSW index yield 100% parity with sequential baseline")
  void twentyThreadsConcurrentSearchMatchesSequentialBaseline() throws Exception {
    Random rng = new Random(42);
    HnswConfig config = HnswConfig.withSeed(42L);
    HnswIndex hnsw = new HnswIndex(DIMENSION, DistanceMetric.EUCLIDEAN, config);

    // Populate the single index
    for (int i = 0; i < NUM_VECTORS; i++) {
      float[] vec = generateVector(DIMENSION, rng);
      hnsw.insert(1000L + i, vec);
    }

    ManagedIndex managedIndex =
        IndexLifecycleManager.createManaged(
            "concurrent-hnsw",
            hnsw,
            IndexMetadata.of(
                "concurrent-hnsw", "HNSW", DIMENSION, DistanceMetric.EUCLIDEAN, NUM_VECTORS));

    ConcurrentSearchSession session = new ConcurrentSearchSession();

    // Generate 20 distinct query vectors (one for each thread)
    float[][] queryVectors = new float[NUM_THREADS][DIMENSION];
    for (int t = 0; t < NUM_THREADS; t++) {
      queryVectors[t] = generateVector(DIMENSION, rng);
    }

    // Step 1: Compute gold-standard sequential baseline results
    List<List<SearchResult>> sequentialBaselines = new ArrayList<>(NUM_THREADS);
    for (int t = 0; t < NUM_THREADS; t++) {
      List<SearchResult> baseline = session.search(managedIndex, queryVectors[t], K, EF_SEARCH);
      assertThat(baseline).isNotEmpty();
      sequentialBaselines.add(baseline);
    }

    // Step 2: Launch 20 concurrent threads executing against the single index simultaneously
    ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    CyclicBarrier barrier = new CyclicBarrier(NUM_THREADS);

    try {
      // Repeat the concurrent blast 10 times to stress concurrency and visited set reuse
      for (int iteration = 0; iteration < 10; iteration++) {
        List<Callable<List<SearchResult>>> tasks = new ArrayList<>();
        for (int t = 0; t < NUM_THREADS; t++) {
          final int threadIdx = t;
          tasks.add(
              () -> {
                // Synchronize all 20 threads to begin search at the exact same moment
                barrier.await();
                return session.search(managedIndex, queryVectors[threadIdx], K, EF_SEARCH);
              });
        }

        List<Future<List<SearchResult>>> futures = executor.invokeAll(tasks);

        // Step 3: Verify bit-for-bit parity of results against sequential baseline
        for (int t = 0; t < NUM_THREADS; t++) {
          List<SearchResult> concurrentResult = futures.get(t).get();
          List<SearchResult> expected = sequentialBaselines.get(t);

          assertThat(concurrentResult)
              .as("Thread %d results must match sequential baseline in iteration %d", t, iteration)
              .hasSameSizeAs(expected);

          for (int i = 0; i < expected.size(); i++) {
            SearchResult exp = expected.get(i);
            SearchResult act = concurrentResult.get(i);

            assertThat(act.id())
                .as("Thread %d neighbor ID at rank %d must match", t, i)
                .isEqualTo(exp.id());

            assertThat(act.distance())
                .as("Thread %d distance at rank %d must match", t, i)
                .isCloseTo(exp.distance(), within(1e-6f));
          }
        }
      }
    } finally {
      executor.shutdownNow();
    }
  }

  @Test
  @DisplayName(
      "20 concurrent threads querying the SAME OffHeapQuantizedHnswIndex yield 100% parity")
  void twentyThreadsConcurrentSearchOnOffHeapIndexMatchesBaseline() throws Exception {
    Random rng = new Random(1337);
    HnswConfig config = HnswConfig.withSeed(1337L);
    OffHeapQuantizedHnswIndex offHeapIndex =
        new OffHeapQuantizedHnswIndex(DIMENSION, DistanceMetric.EUCLIDEAN, config);

    // Populate the single off-heap index
    for (int i = 0; i < NUM_VECTORS; i++) {
      float[] vec = generateVector(DIMENSION, rng);
      offHeapIndex.insert(5000L + i, vec);
    }

    ManagedIndex managedIndex =
        IndexLifecycleManager.createManaged(
            "concurrent-offheap",
            offHeapIndex,
            IndexMetadata.of(
                "concurrent-offheap",
                "OFFHEAP_QUANTIZED_HNSW",
                DIMENSION,
                DistanceMetric.EUCLIDEAN,
                NUM_VECTORS));

    ConcurrentSearchSession session = new ConcurrentSearchSession();

    float[][] queryVectors = new float[NUM_THREADS][DIMENSION];
    for (int t = 0; t < NUM_THREADS; t++) {
      queryVectors[t] = generateVector(DIMENSION, rng);
    }

    // Step 1: Compute sequential baseline
    List<List<SearchResult>> sequentialBaselines = new ArrayList<>(NUM_THREADS);
    for (int t = 0; t < NUM_THREADS; t++) {
      List<SearchResult> baseline = session.search(managedIndex, queryVectors[t], K, EF_SEARCH);
      assertThat(baseline).isNotEmpty();
      sequentialBaselines.add(baseline);
    }

    // Step 2: 20 concurrent threads querying simultaneously
    ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    CyclicBarrier barrier = new CyclicBarrier(NUM_THREADS);

    try {
      for (int iteration = 0; iteration < 10; iteration++) {
        List<Callable<List<SearchResult>>> tasks = new ArrayList<>();
        for (int t = 0; t < NUM_THREADS; t++) {
          final int threadIdx = t;
          tasks.add(
              () -> {
                barrier.await();
                return session.search(managedIndex, queryVectors[threadIdx], K, EF_SEARCH);
              });
        }

        List<Future<List<SearchResult>>> futures = executor.invokeAll(tasks);

        // Step 3: Verify parity
        for (int t = 0; t < NUM_THREADS; t++) {
          List<SearchResult> concurrentResult = futures.get(t).get();
          List<SearchResult> expected = sequentialBaselines.get(t);

          assertThat(concurrentResult)
              .as(
                  "Off-heap thread %d results must match sequential baseline in iteration %d",
                  t, iteration)
              .hasSameSizeAs(expected);

          for (int i = 0; i < expected.size(); i++) {
            SearchResult exp = expected.get(i);
            SearchResult act = concurrentResult.get(i);

            assertThat(act.id())
                .as("Off-heap thread %d neighbor ID at rank %d must match", t, i)
                .isEqualTo(exp.id());

            assertThat(act.distance())
                .as("Off-heap thread %d distance at rank %d must match", t, i)
                .isCloseTo(exp.distance(), within(1e-6f));
          }
        }
      }
    } finally {
      executor.shutdownNow();
      managedIndex.close();
    }
  }

  private static float[] generateVector(int dim, Random rng) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = rng.nextFloat() * 2.0f - 1.0f;
    }
    return v;
  }
}
