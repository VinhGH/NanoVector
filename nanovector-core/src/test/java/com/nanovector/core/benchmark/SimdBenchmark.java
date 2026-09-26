package com.nanovector.core.benchmark;

import com.nanovector.core.distance.DistanceCalculator;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.storage.VectorStorage;
import com.nanovector.core.util.VectorUtils;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/**
 * Micro- and macro-benchmark suite comparing Scalar vs SIMD (Java Vector API) performance across
 * distance metrics, vector dimensions, and index searches.
 */
public class SimdBenchmark {

  private static final int[] DIMENSIONS = {32, 64, 100, 128, 384, 768, 1536};
  private static final DistanceMetric[] METRICS = {
    DistanceMetric.EUCLIDEAN, DistanceMetric.COSINE, DistanceMetric.DOT_PRODUCT
  };

  private static final int DISTANCE_WARMUP_ROUNDS = 20_000;
  private static final int DISTANCE_BENCHMARK_ROUNDS = 100_000;

  private static final int SEARCH_NUM_VECTORS = 1000;
  private static final int SEARCH_DIMENSION = 128;
  private static final int SEARCH_WARMUP_QUERIES = 1000;
  private static final int SEARCH_BENCHMARK_QUERIES = 5000;
  private static final int SEARCH_K = 10;
  private static final long SEED = 42L;

  @Test
  void runFullSimdBenchmark() {
    System.out.println(
        "================================================================================");
    System.out.println(
        "                 NANOVECTOR SIMD ACCELERATION BENCHMARK SUITE                   ");
    System.out.println(
        "================================================================================");
    System.out.printf(
        "Java Runtime     : %s (%s)%n",
        System.getProperty("java.version"), System.getProperty("java.vendor"));
    System.out.printf(
        "Vector Species   : %s (%d lanes, %d bits)%n",
        FloatVector.SPECIES_PREFERRED,
        FloatVector.SPECIES_PREFERRED.length(),
        FloatVector.SPECIES_PREFERRED.vectorBitSize());
    System.out.printf("Available Cores  : %d%n", Runtime.getRuntime().availableProcessors());
    System.out.println(
        "================================================================================");

    // Part 1: Raw Distance Throughput Across Dimensions and Metrics
    benchmarkRawDistanceThroughput();

    // Part 2: FlatIndex Brute-Force Scan (Scalar vs SIMD)
    benchmarkFlatIndexSearch();

    // Part 3: HnswIndex Graph Traversal (Scalar vs SIMD)
    benchmarkHnswIndexSearch();

    System.out.println(
        "\n================================================================================");
    System.out.println(
        "                          BENCHMARK RUN COMPLETED                               ");
    System.out.println(
        "================================================================================");
  }

  private void benchmarkRawDistanceThroughput() {
    System.out.println("\n--- [PART 1] RAW DISTANCE CALCULATOR THROUGHPUT (Buffer-to-Query) ---");
    System.out.println("Iterations: " + DISTANCE_BENCHMARK_ROUNDS + " calls/measurement");
    System.out.printf(
        "%-12s | %-6s | %-14s | %-14s | %-10s%n",
        "Metric", "Dim", "Scalar (MOps/s)", "SIMD (MOps/s)", "Speedup");
    System.out.println("-------------+--------+----------------+----------------+-----------");

    Random rng = new Random(SEED);

    for (DistanceMetric metric : METRICS) {
      DistanceCalculator scalarCalc = DistanceCalculator.create(metric, false);
      DistanceCalculator simdCalc = DistanceCalculator.create(metric, true);

      for (int dim : DIMENSIONS) {
        // Setup data buffer with 64 vectors to simulate realistic memory locality
        int numBufVecs = 64;
        VectorStorage storage = new VectorStorage(dim, numBufVecs);
        for (int i = 0; i < numBufVecs; i++) {
          float[] v = randomVector(dim, rng);
          if (metric == DistanceMetric.COSINE) {
            v = VectorUtils.normalize(v);
          }
          storage.insert(i, v);
        }
        float[] buffer = storage.vectorBuffer();

        float[] query = randomVector(dim, rng);
        if (metric == DistanceMetric.COSINE) {
          query = VectorUtils.normalize(query);
        }

        // JIT Warmup
        float blackhole = 0.0f;
        for (int i = 0; i < DISTANCE_WARMUP_ROUNDS; i++) {
          int offset = (i % numBufVecs) * dim;
          blackhole += scalarCalc.distance(buffer, offset, query);
          blackhole += simdCalc.distance(buffer, offset, query);
        }

        // Benchmark Scalar
        long startScalar = System.nanoTime();
        for (int i = 0; i < DISTANCE_BENCHMARK_ROUNDS; i++) {
          int offset = (i % numBufVecs) * dim;
          blackhole += scalarCalc.distance(buffer, offset, query);
        }
        long durationScalarNs = System.nanoTime() - startScalar;

        // Benchmark SIMD
        long startSimd = System.nanoTime();
        for (int i = 0; i < DISTANCE_BENCHMARK_ROUNDS; i++) {
          int offset = (i % numBufVecs) * dim;
          blackhole += simdCalc.distance(buffer, offset, query);
        }
        long durationSimdNs = System.nanoTime() - startSimd;

        // Prevent dead code elimination
        if (blackhole == 42.12345f) {
          System.out.println("Anti-optimization anchor");
        }

        double scalarMops =
            (DISTANCE_BENCHMARK_ROUNDS / (durationScalarNs / 1_000_000_000.0)) / 1_000_000.0;
        double simdMops =
            (DISTANCE_BENCHMARK_ROUNDS / (durationSimdNs / 1_000_000_000.0)) / 1_000_000.0;
        double speedup = (double) durationScalarNs / durationSimdNs;

        System.out.printf(
            "%-12s | %-6d | %14.2f | %14.2f | %9.2fx%n",
            metric.name(), dim, scalarMops, simdMops, speedup);
      }
      System.out.println("-------------+--------+----------------+----------------+-----------");
    }
  }

  private void benchmarkFlatIndexSearch() {
    System.out.println("\n--- [PART 2] FLATINDEX (BRUTE-FORCE SCAN) PERFORMANCE ---");
    System.out.printf(
        "Dataset: N=%d, D=%d, Queries=%d, k=%d%n",
        SEARCH_NUM_VECTORS, SEARCH_DIMENSION, SEARCH_BENCHMARK_QUERIES, SEARCH_K);

    Random rng = new Random(SEED);
    List<float[]> dataset = generateDataset(SEARCH_NUM_VECTORS, SEARCH_DIMENSION, rng);
    List<float[]> queries = generateDataset(SEARCH_BENCHMARK_QUERIES, SEARCH_DIMENSION, rng);

    FlatIndex scalarIndex = new FlatIndex(SEARCH_DIMENSION, DistanceMetric.EUCLIDEAN, false);
    FlatIndex simdIndex = new FlatIndex(SEARCH_DIMENSION, DistanceMetric.EUCLIDEAN, true);

    for (int i = 0; i < SEARCH_NUM_VECTORS; i++) {
      scalarIndex.insert(i, dataset.get(i));
      simdIndex.insert(i, dataset.get(i));
    }

    // Warmup
    for (int i = 0; i < SEARCH_WARMUP_QUERIES; i++) {
      scalarIndex.searchKnn(queries.get(i), SEARCH_K);
      simdIndex.searchKnn(queries.get(i), SEARCH_K);
    }

    // Benchmark Scalar FlatIndex
    double[] scalarLatencies = new double[SEARCH_BENCHMARK_QUERIES];
    long scalarTotalStart = System.nanoTime();
    for (int i = 0; i < SEARCH_BENCHMARK_QUERIES; i++) {
      long qStart = System.nanoTime();
      scalarIndex.searchKnn(queries.get(i), SEARCH_K);
      scalarLatencies[i] = (System.nanoTime() - qStart) / 1000.0; // micros
    }
    long scalarTotalNs = System.nanoTime() - scalarTotalStart;

    // Benchmark SIMD FlatIndex
    double[] simdLatencies = new double[SEARCH_BENCHMARK_QUERIES];
    long simdTotalStart = System.nanoTime();
    for (int i = 0; i < SEARCH_BENCHMARK_QUERIES; i++) {
      long qStart = System.nanoTime();
      simdIndex.searchKnn(queries.get(i), SEARCH_K);
      simdLatencies[i] = (System.nanoTime() - qStart) / 1000.0; // micros
    }
    long simdTotalNs = System.nanoTime() - simdTotalStart;

    Arrays.sort(scalarLatencies);
    Arrays.sort(simdLatencies);

    double scalarQps = (SEARCH_BENCHMARK_QUERIES / (scalarTotalNs / 1_000_000_000.0));
    double simdQps = (SEARCH_BENCHMARK_QUERIES / (simdTotalNs / 1_000_000_000.0));
    double speedup = simdQps / scalarQps;

    System.out.printf(
        "%-16s | %-10s | %-10s | %-10s | %-10s | %-10s%n",
        "Index Engine", "QPS", "Avg (μs)", "p50 (μs)", "p95 (μs)", "p99 (μs)");
    System.out.println(
        "-----------------+------------+------------+------------+------------+-----------");
    System.out.printf(
        "%-16s | %10.0f | %10.2f | %10.2f | %10.2f | %10.2f%n",
        "Flat (Scalar)",
        scalarQps,
        (scalarTotalNs / 1000.0) / SEARCH_BENCHMARK_QUERIES,
        scalarLatencies[(int) (SEARCH_BENCHMARK_QUERIES * 0.50)],
        scalarLatencies[(int) (SEARCH_BENCHMARK_QUERIES * 0.95)],
        scalarLatencies[(int) (SEARCH_BENCHMARK_QUERIES * 0.99)]);
    System.out.printf(
        "%-16s | %10.0f | %10.2f | %10.2f | %10.2f | %10.2f%n",
        "Flat (SIMD)",
        simdQps,
        (simdTotalNs / 1000.0) / SEARCH_BENCHMARK_QUERIES,
        simdLatencies[(int) (SEARCH_BENCHMARK_QUERIES * 0.50)],
        simdLatencies[(int) (SEARCH_BENCHMARK_QUERIES * 0.95)],
        simdLatencies[(int) (SEARCH_BENCHMARK_QUERIES * 0.99)]);
    System.out.printf(">>> FlatIndex Brute-Force Scan Speedup: %.2fx%n", speedup);
  }

  private void benchmarkHnswIndexSearch() {
    System.out.println("\n--- [PART 3] HNSW GRAPH SEARCH PERFORMANCE (Scalar vs SIMD) ---");
    System.out.printf(
        "Dataset: N=%d, D=%d, M=16, efConstruction=200, Queries=%d, k=%d%n",
        SEARCH_NUM_VECTORS, SEARCH_DIMENSION, SEARCH_BENCHMARK_QUERIES, SEARCH_K);

    Random rng = new Random(SEED);
    List<float[]> dataset = generateDataset(SEARCH_NUM_VECTORS, SEARCH_DIMENSION, rng);
    List<float[]> queries = generateDataset(SEARCH_BENCHMARK_QUERIES, SEARCH_DIMENSION, rng);

    HnswConfig config = new HnswConfig(16, 32, 200, 50, 1.0 / Math.log(16), SEED);
    HnswIndex scalarHnsw = new HnswIndex(SEARCH_DIMENSION, DistanceMetric.EUCLIDEAN, config, false);
    HnswIndex simdHnsw = new HnswIndex(SEARCH_DIMENSION, DistanceMetric.EUCLIDEAN, config, true);

    // Build indexes
    long buildScalarStart = System.nanoTime();
    for (int i = 0; i < SEARCH_NUM_VECTORS; i++) {
      scalarHnsw.insert(i, dataset.get(i));
    }
    double buildScalarMs = (System.nanoTime() - buildScalarStart) / 1_000_000.0;

    long buildSimdStart = System.nanoTime();
    for (int i = 0; i < SEARCH_NUM_VECTORS; i++) {
      simdHnsw.insert(i, dataset.get(i));
    }
    double buildSimdMs = (System.nanoTime() - buildSimdStart) / 1_000_000.0;

    System.out.printf(
        "Build Time: Scalar = %.2f ms | SIMD = %.2f ms (Build Speedup: %.2fx)%n",
        buildScalarMs, buildSimdMs, buildScalarMs / buildSimdMs);

    int[] efSearchList = {10, 50, 100};

    System.out.printf(
        "%-9s | %-14s | %-10s | %-10s | %-10s | %-10s | %-8s%n",
        "efSearch", "Engine", "QPS", "Avg (μs)", "p50 (μs)", "p99 (μs)", "Speedup");
    System.out.println(
        "----------+----------------+------------+------------+------------+------------+---------");

    for (int efSearch : efSearchList) {
      // Warmup
      for (int i = 0; i < SEARCH_WARMUP_QUERIES; i++) {
        scalarHnsw.searchKnn(queries.get(i), SEARCH_K, efSearch);
        simdHnsw.searchKnn(queries.get(i), SEARCH_K, efSearch);
      }

      // Benchmark Scalar
      double[] scalarLatencies = new double[SEARCH_BENCHMARK_QUERIES];
      long scalarStart = System.nanoTime();
      for (int i = 0; i < SEARCH_BENCHMARK_QUERIES; i++) {
        long qStart = System.nanoTime();
        scalarHnsw.searchKnn(queries.get(i), SEARCH_K, efSearch);
        scalarLatencies[i] = (System.nanoTime() - qStart) / 1000.0;
      }
      long scalarNs = System.nanoTime() - scalarStart;

      // Benchmark SIMD
      double[] simdLatencies = new double[SEARCH_BENCHMARK_QUERIES];
      long simdStart = System.nanoTime();
      for (int i = 0; i < SEARCH_BENCHMARK_QUERIES; i++) {
        long qStart = System.nanoTime();
        simdHnsw.searchKnn(queries.get(i), SEARCH_K, efSearch);
        simdLatencies[i] = (System.nanoTime() - qStart) / 1000.0;
      }
      long simdNs = System.nanoTime() - simdStart;

      Arrays.sort(scalarLatencies);
      Arrays.sort(simdLatencies);

      double scalarQps = SEARCH_BENCHMARK_QUERIES / (scalarNs / 1_000_000_000.0);
      double simdQps = SEARCH_BENCHMARK_QUERIES / (simdNs / 1_000_000_000.0);
      double speedup = simdQps / scalarQps;

      System.out.printf(
          "%-9d | %-14s | %10.0f | %10.2f | %10.2f | %10.2f | %7.2fx%n",
          efSearch,
          "HNSW (Scalar)",
          scalarQps,
          (scalarNs / 1000.0) / SEARCH_BENCHMARK_QUERIES,
          scalarLatencies[(int) (SEARCH_BENCHMARK_QUERIES * 0.50)],
          scalarLatencies[(int) (SEARCH_BENCHMARK_QUERIES * 0.99)],
          1.0);
      System.out.printf(
          "%-9d | %-14s | %10.0f | %10.2f | %10.2f | %10.2f | %7.2fx%n",
          efSearch,
          "HNSW (SIMD)",
          simdQps,
          (simdNs / 1000.0) / SEARCH_BENCHMARK_QUERIES,
          simdLatencies[(int) (SEARCH_BENCHMARK_QUERIES * 0.50)],
          simdLatencies[(int) (SEARCH_BENCHMARK_QUERIES * 0.99)],
          speedup);
      System.out.println(
          "----------+----------------+------------+------------+------------+------------+---------");
    }
  }

  private static float[] randomVector(int dim, Random rng) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = rng.nextFloat() * 2.0f - 1.0f;
    }
    return v;
  }

  private static List<float[]> generateDataset(int count, int dim, Random rng) {
    List<float[]> list = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      list.add(randomVector(dim, rng));
    }
    return list;
  }
}
