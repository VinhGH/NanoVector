package com.nanovector.benchmark.search;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.QuantizedHnswIndex;
import com.nanovector.core.model.SearchResult;
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
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * Empirical benchmark mapping the Recall vs Latency/Throughput Pareto Frontier and evaluating
 * Two-Phase Search with FP32 Re-ranking across beam widths ({@code efSearch} &isin; {10, 20, 50,
 * 100, 200, 400}).
 *
 * <p>Scientific investigation:
 *
 * <ol>
 *   <li><b>Pareto Curve Comparison</b>: Compares FP32 HNSW vs Pure SQ8 HNSW across the full
 *       trade-off space between query latency and empirical Recall@10.
 *   <li><b>Two-Phase Search Efficacy</b>: Quantifies the recall recovery achieved by re-ranking SQ8
 *       ADC candidates using exact FP32 Euclidean distances against raw vector buffers.
 *   <li><b>Candidate Pruning Bound</b>: Distinguishes between <i>recoverable ranking loss</i>
 *       (corrected by Phase 2 re-ranking) and <i>unrecoverable graph exploration loss</i> (true
 *       neighbors pruned during HNSW routing and absent from the {@code efSearch} candidate list).
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
public class ParetoFrontierRerankBenchmark {

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
  private HnswIndex jmhFp32Index;
  private QuantizedHnswIndex jmhPureSq8Index;
  private VectorStorage jmhRawStorage;
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
    this.jmhFp32Index = new HnswIndex(DEFAULT_DIMENSION, DEFAULT_METRIC, config, vectorCount, true);
    this.jmhPureSq8Index =
        new QuantizedHnswIndex(DEFAULT_DIMENSION, DEFAULT_METRIC, config, vectorCount, true);
    this.jmhRawStorage = new VectorStorage(DEFAULT_DIMENSION, vectorCount);

    for (int i = 0; i < vectorCount; i++) {
      jmhFp32Index.insert(i, dataset[i]);
      jmhPureSq8Index.insert(i, dataset[i]);
      jmhRawStorage.insert(i, dataset[i]);
    }
  }

  @Benchmark
  public void searchFp32(Blackhole bh) {
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      bh.consume(jmhFp32Index.searchKnn(rawQueries[q], DEFAULT_K, jmhEfSearch));
    }
  }

  @Benchmark
  public void searchPureSq8(Blackhole bh) {
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      bh.consume(jmhPureSq8Index.searchKnn(rawQueries[q], DEFAULT_K, jmhEfSearch));
    }
  }

  @Benchmark
  public void searchSq8WithRerank(Blackhole bh) {
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      bh.consume(
          jmhPureSq8Index.searchKnnWithRerank(
              rawQueries[q], DEFAULT_K, jmhEfSearch, jmhRawStorage));
    }
  }

  /** Result record capturing Pareto trade-off and re-ranking metrics for a single efSearch. */
  public record ParetoStepResult(
      int efSearch,
      int vectorCount,
      int dimension,
      int k,
      // Accuracy / Recall metrics
      double fp32Recall,
      double sq8Recall,
      double candidateRecall,
      double rerankedRecall,
      double recallRecovered,
      double unrecoverableLoss,
      // Throughput (QPS)
      double fp32Qps,
      double sq8Qps,
      double rerankedQps,
      // Latency Distribution (us)
      double fp32MeanLatencyUs,
      double fp32P50LatencyUs,
      double fp32P95LatencyUs,
      double fp32P99LatencyUs,
      double sq8MeanLatencyUs,
      double sq8P50LatencyUs,
      double sq8P95LatencyUs,
      double sq8P99LatencyUs,
      double rerankedMeanLatencyUs,
      double rerankedP50LatencyUs,
      double rerankedP95LatencyUs,
      double rerankedP99LatencyUs,
      double rerankOverheadUs) {

    public String toParetoMarkdownRow() {
      return String.format(
          "| %,8d | %11.2f%% | %,12.1f | %11.2f | %10.2f%% | %,11.1f | %10.2f | %14.2f%% | %,15.1f | %14.2f |",
          efSearch,
          fp32Recall * 100.0,
          fp32Qps,
          fp32MeanLatencyUs,
          sq8Recall * 100.0,
          sq8Qps,
          sq8MeanLatencyUs,
          rerankedRecall * 100.0,
          rerankedQps,
          rerankedMeanLatencyUs);
    }

    public String toRerankDeepDiveMarkdownRow() {
      return String.format(
          "| %,8d | %15.2f%% | %19.2f%% | %16.2f%% | %14.2f%% | %18.2f%% | %14.2f us |",
          efSearch,
          sq8Recall * 100.0,
          candidateRecall * 100.0,
          rerankedRecall * 100.0,
          recallRecovered * 100.0,
          unrecoverableLoss * 100.0,
          rerankOverheadUs);
    }

    public String toLatencyDistributionMarkdownRow() {
      return String.format(
          "| %,8d | %8.1f / %8.1f / %8.1f | %8.1f / %8.1f / %8.1f | %10.1f / %10.1f / %10.1f |",
          efSearch,
          fp32P50LatencyUs,
          fp32P95LatencyUs,
          fp32P99LatencyUs,
          sq8P50LatencyUs,
          sq8P95LatencyUs,
          sq8P99LatencyUs,
          rerankedP50LatencyUs,
          rerankedP95LatencyUs,
          rerankedP99LatencyUs);
    }
  }

  /**
   * Evaluates the Pareto Frontier and Two-Phase Re-ranking for a specific scale and efSearch value.
   */
  public static ParetoStepResult evaluateStep(
      int n,
      int dimension,
      int k,
      int efSearch,
      int numQueries,
      long dataSeed,
      long querySeed,
      FlatIndex oracle,
      HnswIndex fp32Index,
      QuantizedHnswIndex pureSq8Index,
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

    // 2. Measure Recall: FP32, Pure SQ8, Candidates, Re-ranked
    int totalFp32Hits = 0;
    int totalSq8Hits = 0;
    int totalCandidateHits = 0;
    int totalRerankedHits = 0;

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

      // Pure SQ8 search
      List<SearchResult> sq8Res = pureSq8Index.searchKnn(query, k, efSearch);
      for (SearchResult r : sq8Res) {
        if (gt.contains(r.id())) {
          totalSq8Hits++;
        }
      }

      // Candidate retrieval
      List<SearchResult> candidates = pureSq8Index.searchKnnCandidates(query, efSearch);
      for (long gtId : gt) {
        for (SearchResult c : candidates) {
          if (c.id() == gtId) {
            totalCandidateHits++;
            break;
          }
        }
      }

      // Re-ranked search
      List<SearchResult> rerankedRes =
          pureSq8Index.searchKnnWithRerank(query, k, efSearch, rawStorage);
      for (SearchResult r : rerankedRes) {
        if (gt.contains(r.id())) {
          totalRerankedHits++;
        }
      }
    }

    double fp32Recall = (double) totalFp32Hits / (numQueries * k);
    double sq8Recall = (double) totalSq8Hits / (numQueries * k);
    double candidateRecall = (double) totalCandidateHits / (numQueries * k);
    double rerankedRecall = (double) totalRerankedHits / (numQueries * k);
    double recallRecovered = rerankedRecall - sq8Recall;
    double unrecoverableLoss = 1.0 - candidateRecall;

    // 3. Timed search & latency distributions (5 passes)
    int passes = 5;
    int totalOps = numQueries * passes;
    long[] fp32LatenciesNs = new long[totalOps];
    long[] sq8LatenciesNs = new long[totalOps];
    long[] rerankLatenciesNs = new long[totalOps];

    // Warmup
    for (int q = 0; q < Math.min(32, numQueries); q++) {
      fp32Index.searchKnn(queries[q], k, efSearch);
      pureSq8Index.searchKnn(queries[q], k, efSearch);
      pureSq8Index.searchKnnWithRerank(queries[q], k, efSearch, rawStorage);
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
    long startSq8 = System.nanoTime();
    for (int p = 0; p < passes; p++) {
      for (int q = 0; q < numQueries; q++) {
        long t0 = System.nanoTime();
        pureSq8Index.searchKnn(queries[q], k, efSearch);
        sq8LatenciesNs[idx++] = System.nanoTime() - t0;
      }
    }
    long totalSq8Ns = System.nanoTime() - startSq8;

    idx = 0;
    long startRerank = System.nanoTime();
    for (int p = 0; p < passes; p++) {
      for (int q = 0; q < numQueries; q++) {
        long t0 = System.nanoTime();
        pureSq8Index.searchKnnWithRerank(queries[q], k, efSearch, rawStorage);
        rerankLatenciesNs[idx++] = System.nanoTime() - t0;
      }
    }
    long totalRerankNs = System.nanoTime() - startRerank;

    double fp32Qps = (double) totalOps / (totalFp32Ns / 1_000_000_000.0);
    double sq8Qps = (double) totalOps / (totalSq8Ns / 1_000_000_000.0);
    double rerankedQps = (double) totalOps / (totalRerankNs / 1_000_000_000.0);

    Arrays.sort(fp32LatenciesNs);
    Arrays.sort(sq8LatenciesNs);
    Arrays.sort(rerankLatenciesNs);

    double fp32Mean = (double) totalFp32Ns / totalOps / 1000.0;
    double fp32P50 = fp32LatenciesNs[(int) (totalOps * 0.50)] / 1000.0;
    double fp32P95 = fp32LatenciesNs[(int) (totalOps * 0.95)] / 1000.0;
    double fp32P99 = fp32LatenciesNs[(int) (totalOps * 0.99)] / 1000.0;

    double sq8Mean = (double) totalSq8Ns / totalOps / 1000.0;
    double sq8P50 = sq8LatenciesNs[(int) (totalOps * 0.50)] / 1000.0;
    double sq8P95 = sq8LatenciesNs[(int) (totalOps * 0.95)] / 1000.0;
    double sq8P99 = sq8LatenciesNs[(int) (totalOps * 0.99)] / 1000.0;

    double rerankMean = (double) totalRerankNs / totalOps / 1000.0;
    double rerankP50 = rerankLatenciesNs[(int) (totalOps * 0.50)] / 1000.0;
    double rerankP95 = rerankLatenciesNs[(int) (totalOps * 0.95)] / 1000.0;
    double rerankP99 = rerankLatenciesNs[(int) (totalOps * 0.99)] / 1000.0;

    double rerankOverhead = Math.max(0.0, rerankMean - sq8Mean);

    return new ParetoStepResult(
        efSearch,
        n,
        dimension,
        k,
        fp32Recall,
        sq8Recall,
        candidateRecall,
        rerankedRecall,
        recallRecovered,
        unrecoverableLoss,
        fp32Qps,
        sq8Qps,
        rerankedQps,
        fp32Mean,
        fp32P50,
        fp32P95,
        fp32P99,
        sq8Mean,
        sq8P50,
        sq8P95,
        sq8P99,
        rerankMean,
        rerankP50,
        rerankP95,
        rerankP99,
        rerankOverhead);
  }

  /** Runs the Pareto Frontier sweep across efSearch values for a specific dataset scale. */
  public static List<ParetoStepResult> runParetoSweep(int n, int dimension, int[] efSearchValues) {
    System.out.printf(
        "Preparing Pareto Frontier benchmark for Scale N = %,d (D=%d)...\n", n, dimension);
    System.out.flush();

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

    // Build Ground Truth Oracle
    FlatIndex oracle = new FlatIndex(dimension, DEFAULT_METRIC, n, true);
    for (int i = 0; i < n; i++) {
      oracle.insert(i, dataset[i]);
    }

    // Build FP32 HNSW & Pure SQ8 HNSW
    HnswConfig config = HnswConfig.withSeed(DEFAULT_DATA_SEED);
    HnswIndex fp32Index = new HnswIndex(dimension, DEFAULT_METRIC, config, n, true);
    QuantizedHnswIndex pureSq8Index =
        new QuantizedHnswIndex(dimension, DEFAULT_METRIC, config, n, true);
    VectorStorage rawStorage = new VectorStorage(dimension, n);

    for (int i = 0; i < n; i++) {
      fp32Index.insert(i, dataset[i]);
      pureSq8Index.insert(i, dataset[i]);
      rawStorage.insert(i, dataset[i]);
    }

    List<ParetoStepResult> results = new ArrayList<>();
    for (int ef : efSearchValues) {
      System.out.printf("  Evaluating efSearch = %3d...\n", ef);
      System.out.flush();
      ParetoStepResult step =
          evaluateStep(
              n,
              dimension,
              DEFAULT_K,
              ef,
              DEFAULT_NUM_QUERIES,
              DEFAULT_DATA_SEED,
              DEFAULT_QUERY_SEED,
              oracle,
              fp32Index,
              pureSq8Index,
              rawStorage,
              queries);
      results.add(step);
      System.out.printf(
          "    efSearch=%3d | FP32: %.2f%% (%,.1f QPS) | Pure SQ8: %.2f%% (%,.1f QPS) | Re-ranked: %.2f%% (%,.1f QPS) | Cand Recall: %.2f%%\n",
          ef,
          step.fp32Recall() * 100.0,
          step.fp32Qps(),
          step.sq8Recall() * 100.0,
          step.sq8Qps(),
          step.rerankedRecall() * 100.0,
          step.rerankedQps(),
          step.candidateRecall() * 100.0);
      System.out.flush();
    }
    return results;
  }

  public static void printReportTables(List<ParetoStepResult> results) {
    if (results.isEmpty()) return;
    int n = results.get(0).vectorCount();
    int d = results.get(0).dimension();

    System.out.println(
        "====================================================================================================================");
    System.out.printf(
        "          PHASE 6C: PARETO FRONTIER & TWO-PHASE RE-RANKING STUDY (N=%,d, D=%d, k=10)\n",
        n, d);
    System.out.println(
        "====================================================================================================================");

    System.out.println("### Table 1: Pareto Frontier: Recall@10 vs Throughput & Mean Latency");
    System.out.println(
        "| efSearch | FP32 Recall |   FP32 QPS   | FP32 Lat (us) | SQ8 Recall |   SQ8 QPS    | SQ8 Lat (us) | Rerank Recall |   Rerank QPS   | Rerank Lat (us) |");
    System.out.println(
        "|---------:|------------:|-------------:|--------------:|-----------:|-------------:|-------------:|--------------:|---------------:|----------------:|");
    for (ParetoStepResult r : results) {
      System.out.println(r.toParetoMarkdownRow());
    }
    System.out.println();

    System.out.println("### Table 2: Two-Phase Re-ranking Breakdown: Candidate vs Final Recall");
    System.out.println(
        "| efSearch | Pure SQ8 Recall | Candidate Recall@10 | Re-ranked Recall | Recall Recover | Unrecoverable Loss | Re-rank Overhead |");
    System.out.println(
        "|---------:|----------------:|--------------------:|-----------------:|---------------:|-------------------:|-----------------:|");
    for (ParetoStepResult r : results) {
      System.out.println(r.toRerankDeepDiveMarkdownRow());
    }
    System.out.println();

    System.out.println("### Table 3: Latency Distribution: P50 / P95 / P99 (us)");
    System.out.println(
        "| efSearch |  FP32 (P50/P95/P99 us)  |   SQ8 (P50/P95/P99 us)  | Rerank (P50/P95/P99 us)  |");
    System.out.println(
        "|---------:|------------------------:|------------------------:|-------------------------:|");
    for (ParetoStepResult r : results) {
      System.out.println(r.toLatencyDistributionMarkdownRow());
    }
    System.out.println(
        "====================================================================================================================\n");
  }

  public static void main(String[] args) throws RunnerException {
    if (args.length > 0 && "--jmh".equals(args[0])) {
      Options opt =
          new OptionsBuilder()
              .include(ParetoFrontierRerankBenchmark.class.getSimpleName())
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

    List<ParetoStepResult> results = runParetoSweep(n, DEFAULT_DIMENSION, DEFAULT_EF_SEARCH_SWEEP);
    printReportTables(results);
  }
}
