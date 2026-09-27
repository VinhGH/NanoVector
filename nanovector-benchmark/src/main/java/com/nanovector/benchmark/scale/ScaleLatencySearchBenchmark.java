package com.nanovector.benchmark.scale;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.model.SearchResult;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * JMH Benchmark evaluating search throughput and latency scaling across dataset sizes ($N \in
 * \{1\text{K}, 10\text{K}, 50\text{K}, 100\text{K}\}$ at dimension $D=128$) comparing brute-force
 * {@link FlatIndex} with approximate graph search {@link HnswIndex}.
 *
 * <p>Key experimental controls and objectives:
 *
 * <ul>
 *   <li>Same dataset ($D=128$, uniform synthetic vectors, seed 42) for both indexes at each scale.
 *   <li>Same query set (128 random query vectors, seed 12345) cycled through per search.
 *   <li>SIMD distance acceleration enabled across both indexes.
 *   <li>HNSW configuration: $M=16, M_0=32, efConstruction=200, efSearch=50, seed=42$.
 *   <li>Measures empirical degradation behavior as scale increases 100-fold ($1\text{K} \to
 *       100\text{K}$).
 *   <li>Hypothesis on working set: Increasing $N$ expands raw vector payload from 0.49 MiB to 48.83
 *       MiB. FlatIndex requires sequential full-scan of the entire working set per query,
 *       potentially exceeding effective cache capacity and increasing memory access latency;
 *       whereas HNSW traverses localized graph neighborhoods, evaluating only a fraction of vectors
 *       per query.
 *   <li>Measures empirical Recall@10 of HNSW against exact Flat oracle at each scale.
 * </ul>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 3, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(
    value = 1,
    jvmArgsAppend = {"--add-modules", "jdk.incubator.vector"})
public class ScaleLatencySearchBenchmark {

  private static final int DIMENSION = 128;
  private static final DistanceMetric METRIC = DistanceMetric.EUCLIDEAN;
  private static final int K = 10;
  private static final int EF_SEARCH = 50;
  private static final int NUM_QUERIES = 128;
  private static final int QUERY_MASK = NUM_QUERIES - 1;

  @Param({"1000", "10000", "50000", "100000"})
  private int vectorCount;

  private FlatIndex flatIndex;
  private HnswIndex hnswIndex;
  private float[][] queries;
  private int queryIndex;
  private double measuredRecall;
  private double workingSetMiB;

  @Setup
  @SuppressWarnings("unchecked")
  public void setup() {
    Random dataRng = new Random(42L);
    float[][] dataset = new float[vectorCount][DIMENSION];
    for (int i = 0; i < vectorCount; i++) {
      for (int d = 0; d < DIMENSION; d++) {
        dataset[i][d] = dataRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    Random queryRng = new Random(12345L);
    this.queries = new float[NUM_QUERIES][DIMENSION];
    for (int q = 0; q < NUM_QUERIES; q++) {
      for (int d = 0; d < DIMENSION; d++) {
        this.queries[q][d] = queryRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    this.flatIndex = new FlatIndex(DIMENSION, METRIC, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      this.flatIndex.insert(i, dataset[i]);
    }

    HnswConfig config = HnswConfig.withSeed(42L).withEfSearch(EF_SEARCH);
    this.hnswIndex = new HnswIndex(DIMENSION, METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      this.hnswIndex.insert(i, dataset[i]);
    }

    this.queryIndex = 0;
    this.workingSetMiB = (double) vectorCount * DIMENSION * Float.BYTES / (1024.0 * 1024.0);

    // 1. Compute exact Ground Truth Top-10 Oracle via FlatIndex for the query set
    Set<Long>[] groundTruths = new Set[NUM_QUERIES];
    for (int q = 0; q < NUM_QUERIES; q++) {
      List<SearchResult> exactResults = flatIndex.searchKnn(queries[q], K);
      Set<Long> gtIds = new HashSet<>(K);
      for (SearchResult r : exactResults) {
        gtIds.add(r.id());
      }
      groundTruths[q] = gtIds;
    }

    // 2. Compute empirical Recall@10 of HNSW against Flat oracle
    int totalHits = 0;
    for (int q = 0; q < NUM_QUERIES; q++) {
      List<SearchResult> hnswResults = hnswIndex.searchKnn(queries[q], K, EF_SEARCH);
      Set<Long> gt = groundTruths[q];
      for (SearchResult r : hnswResults) {
        if (gt.contains(r.id())) {
          totalHits++;
        }
      }
    }
    this.measuredRecall = (double) totalHits / (NUM_QUERIES * K);

    // Sanity verification
    List<SearchResult> flatCheck = this.flatIndex.searchKnn(this.queries[0], K);
    List<SearchResult> hnswCheck = this.hnswIndex.searchKnn(this.queries[0], K, EF_SEARCH);
    if (flatCheck.size() != K || hnswCheck.size() != K) {
      throw new IllegalStateException(
          String.format(
              "Search sanity check failed: flatCount=%d, hnswCount=%d, expected=%d",
              flatCheck.size(), hnswCheck.size(), K));
    }

    System.out.printf(
        "[Scale Benchmark Setup] N=%6d | Raw Payload=%6.2f MiB | HNSW Recall@10=%6.2f%%%n",
        vectorCount, workingSetMiB, measuredRecall * 100.0);
  }

  @Benchmark
  public void searchFlat(Blackhole bh) {
    int idx = (queryIndex++) & QUERY_MASK;
    List<SearchResult> results = flatIndex.searchKnn(queries[idx], K);
    bh.consume(results);
  }

  @Benchmark
  public void searchHnsw(Blackhole bh) {
    int idx = (queryIndex++) & QUERY_MASK;
    List<SearchResult> results = hnswIndex.searchKnn(queries[idx], K, EF_SEARCH);
    bh.consume(results);
  }

  /** Record encapsulating empirical latency percentiles, throughput, and search accuracy. */
  public record LatencyProfile(
      String indexType,
      int vectorCount,
      double workingSetMiB,
      double throughputOpsPerSec,
      double meanLatencyUs,
      double p50LatencyUs,
      double p90LatencyUs,
      double p99LatencyUs,
      double p999LatencyUs,
      double maxLatencyUs,
      double recallAt10) {

    public String toMarkdownRow(double flatMeanLatencyUs) {
      double speedup = flatMeanLatencyUs > 0 ? flatMeanLatencyUs / meanLatencyUs : 1.0;
      return String.format(
          "| %-9s | %7d | %13.2f | %12.1f | %14.2f | %11.2f | %11.2f | %11.2f | %10.2f%% | %9.2fx |",
          indexType,
          vectorCount,
          workingSetMiB,
          throughputOpsPerSec,
          meanLatencyUs,
          p50LatencyUs,
          p90LatencyUs,
          p99LatencyUs,
          recallAt10 * 100.0,
          speedup);
    }
  }

  /**
   * Measures latency distribution (P50, P90, P99, max) and throughput across repeated query
   * samples.
   *
   * @param isFlat whether to profile FlatIndex (true) or HnswIndex (false)
   * @param numWarmup number of warmup queries
   * @param numSamples number of measurement queries
   * @return comprehensive latency profile
   */
  public LatencyProfile profileIndex(boolean isFlat, int numWarmup, int numSamples) {
    for (int i = 0; i < numWarmup; i++) {
      float[] q = queries[i & QUERY_MASK];
      List<SearchResult> r =
          isFlat ? flatIndex.searchKnn(q, K) : hnswIndex.searchKnn(q, K, EF_SEARCH);
      if (r.isEmpty()) {
        throw new IllegalStateException("Empty search result during warmup");
      }
    }

    long[] latenciesNs = new long[numSamples];
    long totalStart = System.nanoTime();
    for (int i = 0; i < numSamples; i++) {
      float[] q = queries[i & QUERY_MASK];
      long start = System.nanoTime();
      List<SearchResult> r =
          isFlat ? flatIndex.searchKnn(q, K) : hnswIndex.searchKnn(q, K, EF_SEARCH);
      long end = System.nanoTime();
      latenciesNs[i] = end - start;
      if (r.isEmpty()) {
        throw new IllegalStateException("Empty search result during measurement");
      }
    }
    long totalEnd = System.nanoTime();

    Arrays.sort(latenciesNs);
    double p50 = latenciesNs[(int) (numSamples * 0.50)] / 1000.0;
    double p90 = latenciesNs[(int) (numSamples * 0.90)] / 1000.0;
    double p99 = latenciesNs[(int) (numSamples * 0.99)] / 1000.0;
    double p999 = latenciesNs[Math.min(numSamples - 1, (int) (numSamples * 0.999))] / 1000.0;
    double max = latenciesNs[numSamples - 1] / 1000.0;

    long sumNs = 0;
    for (long l : latenciesNs) {
      sumNs += l;
    }
    double meanUs = (double) sumNs / numSamples / 1000.0;
    double totalSeconds = (totalEnd - totalStart) / 1_000_000_000.0;
    double throughput = numSamples / totalSeconds;
    double recall = isFlat ? 1.0 : measuredRecall;

    return new LatencyProfile(
        isFlat ? "FlatIndex" : "HnswIndex",
        vectorCount,
        workingSetMiB,
        throughput,
        meanUs,
        p50,
        p90,
        p99,
        p999,
        max,
        recall);
  }

  /**
   * Executes an end-to-end latency profiling sweep across specified dataset scales.
   *
   * @param scales array of dataset sizes to evaluate
   */
  public static void runScaleSweep(int[] scales) {
    System.out.println(
        "=========================================================================================================");
    System.out.println(
        "                         NANOVECTOR SCALE & SEARCH LATENCY PROFILING SWEEP                               ");
    System.out.println(
        "=========================================================================================================");
    System.out.println(
        "| Index     | Scale N | Raw Size (MB) | Throughput (QPS) | Mean Lat (us)  | P50 Lat (us)| P90 Lat (us)| P99 Lat (us)| Recall@10  | Speedup   |");
    System.out.println(
        "|:----------|--------:|--------------:|-----------------:|---------------:|------------:|------------:|------------:|-----------:|----------:|");

    for (int n : scales) {
      ScaleLatencySearchBenchmark benchmark = new ScaleLatencySearchBenchmark();
      benchmark.initForTest(n);

      int numSamples = Math.min(5000, Math.max(500, 500000 / n));
      int numWarmup = Math.min(500, numSamples / 5);

      LatencyProfile flatProf = benchmark.profileIndex(true, numWarmup, numSamples);
      LatencyProfile hnswProf = benchmark.profileIndex(false, numWarmup, numSamples);

      System.out.println(flatProf.toMarkdownRow(flatProf.meanLatencyUs()));
      System.out.println(hnswProf.toMarkdownRow(flatProf.meanLatencyUs()));
    }
    System.out.println(
        "=========================================================================================================");
  }

  // Package-private accessors for unit tests
  void initForTest(int count) {
    this.vectorCount = count;
    setup();
  }

  FlatIndex flatIndex() {
    return flatIndex;
  }

  HnswIndex hnswIndex() {
    return hnswIndex;
  }

  double measuredRecall() {
    return measuredRecall;
  }

  double workingSetMiB() {
    return workingSetMiB;
  }

  public static void main(String[] args) throws RunnerException {
    if (args.length > 0 && "--latency-profile".equals(args[0])) {
      runScaleSweep(new int[] {1000, 10000, 50000, 100000});
      return;
    }
    Options opt =
        new OptionsBuilder()
            .include(ScaleLatencySearchBenchmark.class.getSimpleName())
            .forks(1)
            .build();
    new Runner(opt).run();
  }
}
