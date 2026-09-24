package com.nanovector.core.benchmark;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.model.SearchResult;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Microbenchmark measuring HNSW build time, search throughput, latency percentiles, and memory
 * allocation rates before and after performance hardening.
 */
public class HnswBenchmark {

  private static final int NUM_VECTORS = 1000;
  private static final int DIMENSION = 128;
  private static final int WARMUP_QUERIES = 1000;
  private static final int BENCHMARK_QUERIES = 5000;
  private static final int K = 10;
  private static final long SEED = 42L;

  @Test
  void runFullBenchmark() {
    System.out.println(
        "================================================================================");
    System.out.println(
        "                     NANOVECTOR HNSW PERFORMANCE BENCHMARK                      ");
    System.out.println(
        "================================================================================");
    System.out.printf(
        "Dataset: N=%d, D=%d, Metric=EUCLIDEAN, Seed=%d%n", NUM_VECTORS, DIMENSION, SEED);
    System.out.printf("HNSW Config: M=16, M0=32, efConstruction=200%n");
    System.out.printf(
        "Search: k=%d, Warmup=%d queries, Benchmark=%d queries%n",
        K, WARMUP_QUERIES, BENCHMARK_QUERIES);
    System.out.println(
        "--------------------------------------------------------------------------------");

    // 1. Generate dataset
    Random rand = new Random(SEED);
    List<float[]> vectors = new ArrayList<>(NUM_VECTORS);
    for (int i = 0; i < NUM_VECTORS; i++) {
      float[] vec = new float[DIMENSION];
      for (int d = 0; d < DIMENSION; d++) {
        vec[d] = rand.nextFloat();
      }
      vectors.add(vec);
    }

    List<float[]> queries = new ArrayList<>(BENCHMARK_QUERIES);
    for (int i = 0; i < BENCHMARK_QUERIES; i++) {
      float[] q = new float[DIMENSION];
      for (int d = 0; d < DIMENSION; d++) {
        q[d] = rand.nextFloat();
      }
      queries.add(q);
    }

    // 2. Measure HNSW Index Build
    System.gc();
    long allocatedBytesBeforeBuild = getThreadAllocatedBytes();
    long gcCountBeforeBuild = getGcCount();
    long gcTimeBeforeBuild = getGcTime();

    long buildStart = System.nanoTime();
    HnswConfig config = new HnswConfig(16, 32, 200, 50, 1.0 / Math.log(16), SEED);
    HnswIndex hnswIndex = new HnswIndex(DIMENSION, DistanceMetric.EUCLIDEAN, config);
    for (int i = 0; i < NUM_VECTORS; i++) {
      hnswIndex.insert(i, vectors.get(i));
    }
    long buildEnd = System.nanoTime();

    long allocatedBytesAfterBuild = getThreadAllocatedBytes();
    long gcCountAfterBuild = getGcCount();
    long gcTimeAfterBuild = getGcTime();

    double buildTimeMs = (buildEnd - buildStart) / 1_000_000.0;
    double allocatedMbBuild =
        (allocatedBytesAfterBuild - allocatedBytesBeforeBuild) / (1024.0 * 1024.0);

    System.out.println("\n[1] INDEX BUILD PERFORMANCE:");
    System.out.printf("  Build Time           : %.2f ms%n", buildTimeMs);
    System.out.printf("  Thread Allocated     : %.2f MB%n", allocatedMbBuild);
    System.out.printf("  GC Count during build: %d%n", (gcCountAfterBuild - gcCountBeforeBuild));
    System.out.printf("  GC Time during build : %d ms%n", (gcTimeAfterBuild - gcTimeBeforeBuild));

    // 3. Warmup JIT for search
    for (int i = 0; i < WARMUP_QUERIES; i++) {
      hnswIndex.searchKnn(queries.get(i % BENCHMARK_QUERIES), K, 50);
    }

    // 4. Measure HNSW Search across efSearch values
    System.out.println("\n[2] HNSW SEARCH PERFORMANCE (Latencies & Throughput):");
    int[] efSearchValues = {10, 20, 50, 100};

    for (int ef : efSearchValues) {
      double[] latenciesUs = new double[BENCHMARK_QUERIES];

      System.gc();
      long allocatedBytesBeforeSearch = getThreadAllocatedBytes();
      long searchStart = System.nanoTime();

      for (int i = 0; i < BENCHMARK_QUERIES; i++) {
        long qStart = System.nanoTime();
        List<SearchResult> res = hnswIndex.searchKnn(queries.get(i), K, ef);
        long qEnd = System.nanoTime();
        latenciesUs[i] = (qEnd - qStart) / 1000.0;
      }

      long searchEnd = System.nanoTime();
      long allocatedBytesAfterSearch = getThreadAllocatedBytes();

      Arrays.sort(latenciesUs);
      double totalSearchTimeMs = (searchEnd - searchStart) / 1_000_000.0;
      double qps = (BENCHMARK_QUERIES / totalSearchTimeMs) * 1000.0;
      double avgLatencyUs = (searchEnd - searchStart) / (1000.0 * BENCHMARK_QUERIES);
      double p50 = latenciesUs[(int) (BENCHMARK_QUERIES * 0.50)];
      double p95 = latenciesUs[(int) (BENCHMARK_QUERIES * 0.95)];
      double p99 = latenciesUs[(int) (BENCHMARK_QUERIES * 0.99)];
      double bytesPerQuery =
          (double) (allocatedBytesAfterSearch - allocatedBytesBeforeSearch) / BENCHMARK_QUERIES;

      System.out.printf(
          "  efSearch=%3d -> QPS: %,8.0f | Avg: %6.2f µs | p50: %6.2f µs | p95: %6.2f µs | p99: %6.2f µs | Alloc/Query: %,6.0f B%n",
          ef, qps, avgLatencyUs, p50, p95, p99, bytesPerQuery);
    }

    // 5. Compare against FlatIndex Baseline
    System.out.println("\n[3] FLATINDEX ORACLE COMPARISON (Exact Exhaustive Scan):");
    FlatIndex flatIndex = new FlatIndex(DIMENSION, DistanceMetric.EUCLIDEAN);
    for (int i = 0; i < NUM_VECTORS; i++) {
      flatIndex.insert(i, vectors.get(i));
    }
    // FlatIndex warmup
    for (int i = 0; i < 500; i++) {
      flatIndex.searchKnn(queries.get(i), K);
    }

    int flatQueryCount = 1000;
    double[] flatLatenciesUs = new double[flatQueryCount];
    long flatStart = System.nanoTime();
    for (int i = 0; i < flatQueryCount; i++) {
      long qStart = System.nanoTime();
      flatIndex.searchKnn(queries.get(i), K);
      long qEnd = System.nanoTime();
      flatLatenciesUs[i] = (qEnd - qStart) / 1000.0;
    }
    long flatEnd = System.nanoTime();

    Arrays.sort(flatLatenciesUs);
    double flatTotalMs = (flatEnd - flatStart) / 1_000_000.0;
    double flatQps = (flatQueryCount / flatTotalMs) * 1000.0;
    double flatAvgUs = (flatEnd - flatStart) / (1000.0 * flatQueryCount);
    double flatP50 = flatLatenciesUs[(int) (flatQueryCount * 0.50)];
    double flatP95 = flatLatenciesUs[(int) (flatQueryCount * 0.95)];
    double flatP99 = flatLatenciesUs[(int) (flatQueryCount * 0.99)];

    System.out.printf(
        "  FlatIndex    -> QPS: %,8.0f | Avg: %6.2f µs | p50: %6.2f µs | p95: %6.2f µs | p99: %6.2f µs%n",
        flatQps, flatAvgUs, flatP50, flatP95, flatP99);

    System.out.println(
        "================================================================================");
  }

  private static long getThreadAllocatedBytes() {
    try {
      java.lang.management.ThreadMXBean bean = ManagementFactory.getThreadMXBean();
      if (bean instanceof com.sun.management.ThreadMXBean sunBean) {
        return sunBean.getThreadAllocatedBytes(Thread.currentThread().threadId());
      }
    } catch (Throwable ignored) {
    }
    return 0;
  }

  private static long getGcCount() {
    long count = 0;
    for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
      long c = gc.getCollectionCount();
      if (c >= 0) count += c;
    }
    return count;
  }

  private static long getGcTime() {
    long time = 0;
    for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
      long t = gc.getCollectionTime();
      if (t >= 0) time += t;
    }
    return time;
  }
}
