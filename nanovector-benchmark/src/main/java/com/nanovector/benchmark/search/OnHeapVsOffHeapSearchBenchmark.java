package com.nanovector.benchmark.search;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.QuantizedHnswIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.offheap.OffHeapQuantizedHnswIndex;
import com.nanovector.core.storage.VectorStorage;
import java.util.ArrayList;
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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * Empirical benchmark measuring query latency, throughput (QPS), accuracy (Recall@10), behavioral
 * parity, and index construction cost across:
 *
 * <ol>
 *   <li><b>FP32 On-Heap HNSW</b>: Standard FP32 vector representation and pointer-based graph.
 *   <li><b>SQ8 On-Heap HNSW</b>: Byte array quantized vector storage and pointer-based graph.
 *   <li><b>SQ8 Off-Heap HNSW</b>: Native {@link java.lang.foreign.MemorySegment} quantized storage
 *       and flattened native multi-layer graph topology.
 *   <li><b>SQ8 Off-Heap Re-ranked HNSW</b>: Two-phase candidate exploration off-heap followed by
 *       exact FP32 re-ranking.
 * </ol>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 3, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(
    value = 1,
    jvmArgsAppend = {"--add-modules", "jdk.incubator.vector"})
public class OnHeapVsOffHeapSearchBenchmark {

  public static final int DEFAULT_DIMENSION = 128;
  public static final DistanceMetric DEFAULT_METRIC = DistanceMetric.EUCLIDEAN;
  public static final int DEFAULT_K = 10;
  public static final int DEFAULT_NUM_QUERIES = 128;
  public static final long DEFAULT_DATA_SEED = 42L;
  public static final long DEFAULT_QUERY_SEED = 12345L;
  public static final int[] DEFAULT_EF_SEARCH_SWEEP = {10, 20, 50, 100, 200, 400};

  @Param({"50", "100", "200"})
  private int jmhEfSearch;

  private int vectorCount = 10000;

  private HnswIndex onHeapFp32Index;
  private QuantizedHnswIndex onHeapSq8Index;
  private OffHeapQuantizedHnswIndex offHeapSq8Index;
  private VectorStorage rawStorage;
  private float[][] rawQueries;

  @Setup
  public void setup() {
    Random dataRng = new Random(DEFAULT_DATA_SEED);
    float[][] dataset = new float[vectorCount][DEFAULT_DIMENSION];
    for (int i = 0; i < vectorCount; i++) {
      for (int d = 0; d < DEFAULT_DIMENSION; d++) {
        dataset[i][d] = dataRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    Random queryRng = new Random(DEFAULT_QUERY_SEED);
    this.rawQueries = new float[DEFAULT_NUM_QUERIES][DEFAULT_DIMENSION];
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      for (int d = 0; d < DEFAULT_DIMENSION; d++) {
        this.rawQueries[q][d] = queryRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    HnswConfig config = HnswConfig.withSeed(DEFAULT_DATA_SEED).withEfSearch(jmhEfSearch);
    this.onHeapFp32Index =
        new HnswIndex(DEFAULT_DIMENSION, DEFAULT_METRIC, config, vectorCount, true);
    this.onHeapSq8Index =
        new QuantizedHnswIndex(DEFAULT_DIMENSION, DEFAULT_METRIC, config, vectorCount, true);
    this.offHeapSq8Index =
        new OffHeapQuantizedHnswIndex(DEFAULT_DIMENSION, DEFAULT_METRIC, config, vectorCount, true);
    this.rawStorage = new VectorStorage(DEFAULT_DIMENSION, vectorCount);

    for (int i = 0; i < vectorCount; i++) {
      onHeapFp32Index.insert(i, dataset[i]);
      onHeapSq8Index.insert(i, dataset[i]);
      offHeapSq8Index.insert(i, dataset[i]);
      rawStorage.insert(i, dataset[i]);
    }
  }

  @TearDown
  public void tearDown() {
    if (offHeapSq8Index != null) {
      offHeapSq8Index.close();
    }
  }

  @Benchmark
  public void searchOnHeapFp32(Blackhole bh) {
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      var res = onHeapFp32Index.searchKnn(rawQueries[q], DEFAULT_K, jmhEfSearch);
      if (bh != null) {
        bh.consume(res);
      }
    }
  }

  @Benchmark
  public void searchOnHeapSq8(Blackhole bh) {
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      var res = onHeapSq8Index.searchKnn(rawQueries[q], DEFAULT_K, jmhEfSearch);
      if (bh != null) {
        bh.consume(res);
      }
    }
  }

  @Benchmark
  public void searchOffHeapSq8(Blackhole bh) {
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      var res = offHeapSq8Index.searchKnn(rawQueries[q], DEFAULT_K, jmhEfSearch);
      if (bh != null) {
        bh.consume(res);
      }
    }
  }

  @Benchmark
  public void searchOffHeapSq8_Rerank(Blackhole bh) {
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      var res =
          offHeapSq8Index.searchKnnWithRerank(rawQueries[q], DEFAULT_K, jmhEfSearch, rawStorage);
      if (bh != null) {
        bh.consume(res);
      }
    }
  }

  /**
   * Result record capturing index construction time and throughput for FP32, On-Heap SQ8, and
   * Off-Heap SQ8.
   */
  public record ConstructionResult(
      int vectorCount,
      int dimension,
      long fp32BuildMs,
      long onHeapSq8BuildMs,
      long offHeapSq8BuildMs,
      double fp32VecPerSec,
      double onHeapSq8VecPerSec,
      double offHeapSq8VecPerSec,
      double offHeapOverheadVsOnHeapSq8,
      double offHeapOverheadVsFp32) {

    public String toMarkdownRow() {
      return String.format(
          "| %,8d | %,12d | %,15.1f | %,14d | %,17.1f | %,15d | %,18.1f | %17.2fx | %13.2fx |",
          vectorCount,
          fp32BuildMs,
          fp32VecPerSec,
          onHeapSq8BuildMs,
          onHeapSq8VecPerSec,
          offHeapSq8BuildMs,
          offHeapSq8VecPerSec,
          offHeapOverheadVsOnHeapSq8,
          offHeapOverheadVsFp32);
    }
  }

  /**
   * Result record capturing search latency, throughput, and recall parity metrics for a given
   * efSearch.
   */
  public record SearchStepResult(
      int efSearch,
      int vectorCount,
      int dimension,
      int k,
      // Accuracy / Parity metrics
      double fp32Recall,
      double onHeapSq8Recall,
      double offHeapSq8Recall,
      double rerankedRecall,
      double parityAgreementRate,
      // Throughput (QPS)
      double fp32Qps,
      double onHeapSq8Qps,
      double offHeapSq8Qps,
      double rerankedQps,
      double offHeapSpeedupVsOnHeapSq8,
      // Latency Distribution (us)
      double fp32MeanUs,
      double fp32P50Us,
      double fp32P95Us,
      double fp32P99Us,
      double onHeapSq8MeanUs,
      double onHeapSq8P50Us,
      double onHeapSq8P95Us,
      double onHeapSq8P99Us,
      double offHeapSq8MeanUs,
      double offHeapSq8P50Us,
      double offHeapSq8P95Us,
      double offHeapSq8P99Us,
      double rerankedMeanUs,
      double rerankedP50Us,
      double rerankedP95Us,
      double rerankedP99Us) {

    public String toSearchLatencyMarkdownRow() {
      return String.format(
          "| %,8d | %8.1f / %8.1f / %8.1f | %,11.1f | %8.1f / %8.1f / %8.1f | %,11.1f | %8.1f / %8.1f / %8.1f | %,12.1f | %15.2fx | %8.1f / %8.1f / %8.1f | %,13.1f |",
          efSearch,
          fp32P50Us,
          fp32P95Us,
          fp32P99Us,
          fp32Qps,
          onHeapSq8P50Us,
          onHeapSq8P95Us,
          onHeapSq8P99Us,
          onHeapSq8Qps,
          offHeapSq8P50Us,
          offHeapSq8P95Us,
          offHeapSq8P99Us,
          offHeapSq8Qps,
          offHeapSpeedupVsOnHeapSq8,
          rerankedP50Us,
          rerankedP95Us,
          rerankedP99Us,
          rerankedQps);
    }

    public String toAccuracyParityMarkdownRow() {
      return String.format(
          "| %,8d | %11.2f%% | %18.2f%% | %19.2f%% | %20.2f%% | %19.2f%% |",
          efSearch,
          fp32Recall * 100.0,
          onHeapSq8Recall * 100.0,
          offHeapSq8Recall * 100.0,
          rerankedRecall * 100.0,
          parityAgreementRate * 100.0);
    }
  }

  /** Complete sweep result bundle containing construction metrics and search step results. */
  public record BenchmarkSweepResult(
      ConstructionResult construction, List<SearchStepResult> searchSteps) {}

  /** Measures index construction performance across FP32, On-Heap SQ8, and Off-Heap SQ8. */
  public static ConstructionResult evaluateConstruction(
      int vectorCount, int dimension, float[][] dataset, HnswConfig config) {
    // 1. FP32 Construction
    long t0 = System.nanoTime();
    HnswIndex fp32 = new HnswIndex(dimension, DEFAULT_METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      fp32.insert(i, dataset[i]);
    }
    long fp32BuildMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0);

    // 2. On-Heap SQ8 Construction
    t0 = System.nanoTime();
    QuantizedHnswIndex onHeapSq8 =
        new QuantizedHnswIndex(dimension, DEFAULT_METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      onHeapSq8.insert(i, dataset[i]);
    }
    long onHeapSq8BuildMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0);

    // 3. Off-Heap SQ8 Construction
    t0 = System.nanoTime();
    OffHeapQuantizedHnswIndex offHeapSq8 =
        new OffHeapQuantizedHnswIndex(dimension, DEFAULT_METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      offHeapSq8.insert(i, dataset[i]);
    }
    long offHeapSq8BuildMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0);
    offHeapSq8.close();

    double fp32Sec = Math.max(0.001, fp32BuildMs / 1000.0);
    double onHeapSq8Sec = Math.max(0.001, onHeapSq8BuildMs / 1000.0);
    double offHeapSq8Sec = Math.max(0.001, offHeapSq8BuildMs / 1000.0);

    double fp32VecPerSec = vectorCount / fp32Sec;
    double onHeapSq8VecPerSec = vectorCount / onHeapSq8Sec;
    double offHeapSq8VecPerSec = vectorCount / offHeapSq8Sec;

    double overheadVsOnHeapSq8 = (double) offHeapSq8BuildMs / Math.max(1, onHeapSq8BuildMs);
    double overheadVsFp32 = (double) offHeapSq8BuildMs / Math.max(1, fp32BuildMs);

    return new ConstructionResult(
        vectorCount,
        dimension,
        fp32BuildMs,
        onHeapSq8BuildMs,
        offHeapSq8BuildMs,
        fp32VecPerSec,
        onHeapSq8VecPerSec,
        offHeapSq8VecPerSec,
        overheadVsOnHeapSq8,
        overheadVsFp32);
  }

  /**
   * Evaluates search accuracy, parity, throughput, and latency distribution for a specific
   * efSearch.
   */
  public static SearchStepResult evaluateSearchStep(
      int n,
      int dimension,
      int k,
      int efSearch,
      int numQueries,
      FlatIndex oracle,
      HnswIndex fp32Index,
      QuantizedHnswIndex onHeapSq8Index,
      OffHeapQuantizedHnswIndex offHeapSq8Index,
      VectorStorage rawStorage,
      float[][] queries) {

    // 1. Compute Ground Truth Top-k oracle results
    @SuppressWarnings("unchecked")
    Set<Long>[] groundTruths = new Set[numQueries];
    for (int q = 0; q < numQueries; q++) {
      List<SearchResult> gt = oracle.searchKnn(queries[q], k);
      Set<Long> gtIds = new HashSet<>(k);
      for (SearchResult r : gt) {
        gtIds.add(r.id());
      }
      groundTruths[q] = gtIds;
    }

    // 2. Measure Recall and On-Heap vs Off-Heap Behavioral Parity
    int totalFp32Hits = 0;
    int totalOnHeapSq8Hits = 0;
    int totalOffHeapSq8Hits = 0;
    int totalRerankedHits = 0;
    double totalParityAgreement = 0.0;

    for (int q = 0; q < numQueries; q++) {
      float[] query = queries[q];
      Set<Long> gt = groundTruths[q];

      // FP32 search
      List<SearchResult> fp32Res = fp32Index.searchKnn(query, k, efSearch);
      for (SearchResult r : fp32Res) {
        if (gt.contains(r.id())) {
          totalFp32Hits++;
        }
      }

      // On-Heap SQ8 search
      List<SearchResult> onHeapSq8Res = onHeapSq8Index.searchKnn(query, k, efSearch);
      Set<Long> onHeapIds = new HashSet<>(k);
      for (SearchResult r : onHeapSq8Res) {
        if (gt.contains(r.id())) {
          totalOnHeapSq8Hits++;
        }
        onHeapIds.add(r.id());
      }

      // Off-Heap SQ8 search
      List<SearchResult> offHeapSq8Res = offHeapSq8Index.searchKnn(query, k, efSearch);
      int agreementHits = 0;
      for (SearchResult r : offHeapSq8Res) {
        if (gt.contains(r.id())) {
          totalOffHeapSq8Hits++;
        }
        if (onHeapIds.contains(r.id())) {
          agreementHits++;
        }
      }
      totalParityAgreement += (double) agreementHits / Math.max(1, k);

      // Off-Heap SQ8 Re-ranked search
      List<SearchResult> rerankedRes =
          offHeapSq8Index.searchKnnWithRerank(query, k, efSearch, rawStorage);
      for (SearchResult r : rerankedRes) {
        if (gt.contains(r.id())) {
          totalRerankedHits++;
        }
      }
    }

    double fp32Recall = (double) totalFp32Hits / (numQueries * k);
    double onHeapSq8Recall = (double) totalOnHeapSq8Hits / (numQueries * k);
    double offHeapSq8Recall = (double) totalOffHeapSq8Hits / (numQueries * k);
    double rerankedRecall = (double) totalRerankedHits / (numQueries * k);
    double parityAgreementRate = totalParityAgreement / numQueries;

    // 3. Timed search & latency distributions (5 passes)
    int passes = 5;
    int totalOps = numQueries * passes;
    long[] fp32LatenciesNs = new long[totalOps];
    long[] onHeapSq8LatenciesNs = new long[totalOps];
    long[] offHeapSq8LatenciesNs = new long[totalOps];
    long[] rerankLatenciesNs = new long[totalOps];

    // Warmup
    for (int q = 0; q < Math.min(32, numQueries); q++) {
      fp32Index.searchKnn(queries[q], k, efSearch);
      onHeapSq8Index.searchKnn(queries[q], k, efSearch);
      offHeapSq8Index.searchKnn(queries[q], k, efSearch);
      offHeapSq8Index.searchKnnWithRerank(queries[q], k, efSearch, rawStorage);
    }

    int idx = 0;
    long startFp32 = System.nanoTime();
    for (int p = 0; p < passes; p++) {
      for (int q = 0; q < numQueries; q++) {
        long t0 = System.nanoTime();
        fp32Index.searchKnn(queries[q], k, efSearch);
        fp32LatenciesNs[idx++] = System.nanoTime() - t0;
      }
    }
    long totalFp32Ns = System.nanoTime() - startFp32;

    idx = 0;
    long startOnHeapSq8 = System.nanoTime();
    for (int p = 0; p < passes; p++) {
      for (int q = 0; q < numQueries; q++) {
        long t0 = System.nanoTime();
        onHeapSq8Index.searchKnn(queries[q], k, efSearch);
        onHeapSq8LatenciesNs[idx++] = System.nanoTime() - t0;
      }
    }
    long totalOnHeapSq8Ns = System.nanoTime() - startOnHeapSq8;

    idx = 0;
    long startOffHeapSq8 = System.nanoTime();
    for (int p = 0; p < passes; p++) {
      for (int q = 0; q < numQueries; q++) {
        long t0 = System.nanoTime();
        offHeapSq8Index.searchKnn(queries[q], k, efSearch);
        offHeapSq8LatenciesNs[idx++] = System.nanoTime() - t0;
      }
    }
    long totalOffHeapSq8Ns = System.nanoTime() - startOffHeapSq8;

    idx = 0;
    long startRerank = System.nanoTime();
    for (int p = 0; p < passes; p++) {
      for (int q = 0; q < numQueries; q++) {
        long t0 = System.nanoTime();
        offHeapSq8Index.searchKnnWithRerank(queries[q], k, efSearch, rawStorage);
        rerankLatenciesNs[idx++] = System.nanoTime() - t0;
      }
    }
    long totalRerankNs = System.nanoTime() - startRerank;

    double fp32Qps = (double) totalOps / (totalFp32Ns / 1_000_000_000.0);
    double onHeapSq8Qps = (double) totalOps / (totalOnHeapSq8Ns / 1_000_000_000.0);
    double offHeapSq8Qps = (double) totalOps / (totalOffHeapSq8Ns / 1_000_000_000.0);
    double rerankedQps = (double) totalOps / (totalRerankNs / 1_000_000_000.0);

    double offHeapSpeedupVsOnHeapSq8 = offHeapSq8Qps / Math.max(1e-9, onHeapSq8Qps);

    Arrays.sort(fp32LatenciesNs);
    Arrays.sort(onHeapSq8LatenciesNs);
    Arrays.sort(offHeapSq8LatenciesNs);
    Arrays.sort(rerankLatenciesNs);

    double fp32Mean = (double) totalFp32Ns / totalOps / 1000.0;
    double fp32P50 = fp32LatenciesNs[(int) (totalOps * 0.50)] / 1000.0;
    double fp32P95 = fp32LatenciesNs[(int) (totalOps * 0.95)] / 1000.0;
    double fp32P99 = fp32LatenciesNs[(int) (totalOps * 0.99)] / 1000.0;

    double onHeapSq8Mean = (double) totalOnHeapSq8Ns / totalOps / 1000.0;
    double onHeapSq8P50 = onHeapSq8LatenciesNs[(int) (totalOps * 0.50)] / 1000.0;
    double onHeapSq8P95 = onHeapSq8LatenciesNs[(int) (totalOps * 0.95)] / 1000.0;
    double onHeapSq8P99 = onHeapSq8LatenciesNs[(int) (totalOps * 0.99)] / 1000.0;

    double offHeapSq8Mean = (double) totalOffHeapSq8Ns / totalOps / 1000.0;
    double offHeapSq8P50 = offHeapSq8LatenciesNs[(int) (totalOps * 0.50)] / 1000.0;
    double offHeapSq8P95 = offHeapSq8LatenciesNs[(int) (totalOps * 0.95)] / 1000.0;
    double offHeapSq8P99 = offHeapSq8LatenciesNs[(int) (totalOps * 0.99)] / 1000.0;

    double rerankMean = (double) totalRerankNs / totalOps / 1000.0;
    double rerankP50 = rerankLatenciesNs[(int) (totalOps * 0.50)] / 1000.0;
    double rerankP95 = rerankLatenciesNs[(int) (totalOps * 0.95)] / 1000.0;
    double rerankP99 = rerankLatenciesNs[(int) (totalOps * 0.99)] / 1000.0;

    return new SearchStepResult(
        efSearch,
        n,
        dimension,
        k,
        fp32Recall,
        onHeapSq8Recall,
        offHeapSq8Recall,
        rerankedRecall,
        parityAgreementRate,
        fp32Qps,
        onHeapSq8Qps,
        offHeapSq8Qps,
        rerankedQps,
        offHeapSpeedupVsOnHeapSq8,
        fp32Mean,
        fp32P50,
        fp32P95,
        fp32P99,
        onHeapSq8Mean,
        onHeapSq8P50,
        onHeapSq8P95,
        onHeapSq8P99,
        offHeapSq8Mean,
        offHeapSq8P50,
        offHeapSq8P95,
        offHeapSq8P99,
        rerankMean,
        rerankP50,
        rerankP95,
        rerankP99);
  }

  /** Runs construction evaluation followed by the search latency/throughput sweep. */
  public static BenchmarkSweepResult runBenchmarkSweep(int n, int dimension, int[] efSearchValues) {
    Random dataRng = new Random(DEFAULT_DATA_SEED);
    float[][] dataset = new float[n][dimension];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dimension; d++) {
        dataset[i][d] = dataRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    Random queryRng = new Random(DEFAULT_QUERY_SEED);
    float[][] queries = new float[DEFAULT_NUM_QUERIES][dimension];
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      for (int d = 0; d < dimension; d++) {
        queries[q][d] = queryRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    HnswConfig config = HnswConfig.withSeed(DEFAULT_DATA_SEED);

    System.out.printf("Evaluating Index Construction Performance (N=%,d, D=%d)...\n", n, dimension);
    System.out.flush();
    ConstructionResult constructionResult = evaluateConstruction(n, dimension, dataset, config);

    System.out.println("Building indices for search query evaluation...");
    System.out.flush();
    FlatIndex oracle = new FlatIndex(dimension, DEFAULT_METRIC, n, true);
    HnswIndex fp32Index = new HnswIndex(dimension, DEFAULT_METRIC, config, n, true);
    QuantizedHnswIndex onHeapSq8Index =
        new QuantizedHnswIndex(dimension, DEFAULT_METRIC, config, n, true);
    OffHeapQuantizedHnswIndex offHeapSq8Index =
        new OffHeapQuantizedHnswIndex(dimension, DEFAULT_METRIC, config, n, true);
    VectorStorage rawStorage = new VectorStorage(dimension, n);

    for (int i = 0; i < n; i++) {
      oracle.insert(i, dataset[i]);
      fp32Index.insert(i, dataset[i]);
      onHeapSq8Index.insert(i, dataset[i]);
      offHeapSq8Index.insert(i, dataset[i]);
      rawStorage.insert(i, dataset[i]);
    }

    List<SearchStepResult> searchSteps = new ArrayList<>();
    try {
      for (int ef : efSearchValues) {
        System.out.printf("  Evaluating search with efSearch = %3d...\n", ef);
        System.out.flush();
        SearchStepResult step =
            evaluateSearchStep(
                n,
                dimension,
                DEFAULT_K,
                ef,
                DEFAULT_NUM_QUERIES,
                oracle,
                fp32Index,
                onHeapSq8Index,
                offHeapSq8Index,
                rawStorage,
                queries);
        searchSteps.add(step);
        System.out.printf(
            "    efSearch=%3d | On-Heap SQ8: %,.1f QPS (%.2f%%) | Off-Heap SQ8: %,.1f QPS (%.2f%%) | Ratio: %.2fx | Parity: %.2f%%\n",
            ef,
            step.onHeapSq8Qps(),
            step.onHeapSq8Recall() * 100.0,
            step.offHeapSq8Qps(),
            step.offHeapSq8Recall() * 100.0,
            step.offHeapSpeedupVsOnHeapSq8(),
            step.parityAgreementRate() * 100.0);
        System.out.flush();
      }
    } finally {
      offHeapSq8Index.close();
    }

    return new BenchmarkSweepResult(constructionResult, searchSteps);
  }

  public static void printReportTables(BenchmarkSweepResult result) {
    ConstructionResult c = result.construction();
    List<SearchStepResult> steps = result.searchSteps();
    if (steps.isEmpty()) return;
    int n = steps.get(0).vectorCount();
    int d = steps.get(0).dimension();

    System.out.println(
        "=========================================================================================================================================");
    System.out.printf(
        "                    PHASE 6D: ON-HEAP VS OFF-HEAP SEARCH & CONSTRUCTION STUDY (N=%,d, D=%d, k=10)\n",
        n, d);
    System.out.println(
        "=========================================================================================================================================");

    System.out.println("### Table 1: Index Construction Performance");
    System.out.println(
        "| Scale (N) | FP32 Build (ms) | FP32 Throughput | On-Heap SQ8 (ms) | On-Heap SQ8 Thrpt | Off-Heap SQ8 (ms) | Off-Heap SQ8 Thrpt | Overhead vs SQ8 | Overhead vs FP32 |");
    System.out.println(
        "|----------:|----------------:|----------------:|-----------------:|------------------:|------------------:|-------------------:|----------------:|-----------------:|");
    System.out.println(c.toMarkdownRow());
    System.out.println();

    System.out.println(
        "### Table 2: Search Latency Distribution (P50/P95/P99 us) & Throughput (QPS)");
    System.out.println(
        "| efSearch | FP32 (P50/P95/P99 us) |   FP32 QPS  | On-Heap SQ8 (P50/P95/P99 us) | On-Heap QPS | Off-Heap SQ8 (P50/P95/P99 us) | Off-Heap QPS | Off-Heap Speedup | Rerank (P50/P95/P99 us) |   Rerank QPS  |");
    System.out.println(
        "|---------:|----------------------:|------------:|-----------------------------:|------------:|------------------------------:|-------------:|-----------------:|------------------------:|--------------:|");
    for (SearchStepResult r : steps) {
      System.out.println(r.toSearchLatencyMarkdownRow());
    }
    System.out.println();

    System.out.println(
        "### Table 3: Accuracy & Behavioral Parity (Recall@10 and Top-10 Agreement)");
    System.out.println(
        "| efSearch | FP32 Recall | On-Heap SQ8 Recall | Off-Heap SQ8 Recall | Re-ranked Recall@10 | On/Off Parity Agreement |");
    System.out.println(
        "|---------:|------------:|-------------------:|--------------------:|--------------------:|------------------------:|");
    for (SearchStepResult r : steps) {
      System.out.println(r.toAccuracyParityMarkdownRow());
    }
    System.out.println(
        "=========================================================================================================================================\n");
  }

  void initForTest(int count) {
    this.vectorCount = count;
    this.jmhEfSearch = 50;
    setup();
  }

  HnswIndex onHeapFp32Index() {
    return onHeapFp32Index;
  }

  QuantizedHnswIndex onHeapSq8Index() {
    return onHeapSq8Index;
  }

  OffHeapQuantizedHnswIndex offHeapSq8Index() {
    return offHeapSq8Index;
  }

  public static void main(String[] args) throws RunnerException {
    if (args.length > 0 && "--jmh".equals(args[0])) {
      Options opt =
          new OptionsBuilder()
              .include(OnHeapVsOffHeapSearchBenchmark.class.getSimpleName())
              .forks(1)
              .build();
      new Runner(opt).run();
      return;
    }

    int n = 10_000;
    if (args.length > 0) {
      try {
        n = Integer.parseInt(args[0]);
      } catch (NumberFormatException ignored) {
      }
    }

    BenchmarkSweepResult result = runBenchmarkSweep(n, DEFAULT_DIMENSION, DEFAULT_EF_SEARCH_SWEEP);
    printReportTables(result);
  }
}
