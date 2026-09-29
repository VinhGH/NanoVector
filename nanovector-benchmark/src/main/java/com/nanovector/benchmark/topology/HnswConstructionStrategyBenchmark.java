package com.nanovector.benchmark.topology;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.QuantizedHnswIndex;
import com.nanovector.core.model.SearchResult;
import java.util.ArrayList;
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
 * Empirical benchmark and topology divergence analyzer evaluating the three HNSW graph construction
 * strategies:
 *
 * <ol>
 *   <li><b>FP32 HNSW (Reference Oracle Graph)</b>: Full-precision FP32 graph construction and FP32
 *       SIMD routing/search.
 *   <li><b>Hybrid SQ8 HNSW (Strategy A)</b>: Constructed with FP32 reference graph topology, but
 *       storing vectors in SQ8 and searching via SQ8 SIMD ADC. Isolates search-time quantization
 *       error from graph routing divergence.
 *   <li><b>Pure SQ8 HNSW (Strategy B)</b>: End-to-end quantized graph construction where neighbor
 *       selection and greedy routing are evaluated entirely via SQ8 SIMD ADC, followed by SQ8 SIMD
 *       ADC search.
 * </ol>
 *
 * <p>Scientific questions addressed:
 *
 * <ul>
 *   <li>Does asymmetric quantized distance distortion degrade edge selection during construction?
 *   <li>How much does Pure SQ8 graph topology diverge from the FP32 reference graph (measured via
 *       Layer-0 Jaccard Edge Similarity)?
 *   <li>Does Pure SQ8 construction preserve graph connectivity invariants (1 connected component,
 *       zero isolated nodes, strict degree bounds)?
 *   <li>How much of the total Recall@10 loss is attributable to quantization distance error alone
 *       versus graph topology divergence?
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
public class HnswConstructionStrategyBenchmark {

  public static final int DEFAULT_DIMENSION = 128;
  public static final DistanceMetric DEFAULT_METRIC = DistanceMetric.EUCLIDEAN;
  public static final int DEFAULT_K = 10;
  public static final int DEFAULT_EF_SEARCH = 50;
  public static final int DEFAULT_NUM_QUERIES = 128;
  public static final long DEFAULT_DATA_SEED = 42L;
  public static final long DEFAULT_QUERY_SEED = 12345L;
  public static final int[] DEFAULT_SCALES = {1_000, 10_000, 50_000, 100_000};

  @Param({"1000"})
  private int vectorCount;

  private float[][] rawDataset;
  private float[][] rawQueries;
  private HnswIndex jmhFp32Index;
  private QuantizedHnswIndex jmhPureSq8Index;

  @Setup
  public void setup() {
    Random dataRng = new Random(DEFAULT_DATA_SEED);
    this.rawDataset = new float[vectorCount][DEFAULT_DIMENSION];
    for (int i = 0; i < vectorCount; i++) {
      for (int d = 0; d < DEFAULT_DIMENSION; d++) {
        this.rawDataset[i][d] = dataRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    Random queryRng = new Random(DEFAULT_QUERY_SEED);
    this.rawQueries = new float[DEFAULT_NUM_QUERIES][DEFAULT_DIMENSION];
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      for (int d = 0; d < DEFAULT_DIMENSION; d++) {
        this.rawQueries[q][d] = queryRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    HnswConfig config = HnswConfig.withSeed(DEFAULT_DATA_SEED).withEfSearch(DEFAULT_EF_SEARCH);
    this.jmhFp32Index = new HnswIndex(DEFAULT_DIMENSION, DEFAULT_METRIC, config, vectorCount, true);
    this.jmhPureSq8Index =
        new QuantizedHnswIndex(DEFAULT_DIMENSION, DEFAULT_METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      this.jmhFp32Index.insert(i, this.rawDataset[i]);
      this.jmhPureSq8Index.insert(i, this.rawDataset[i]);
    }
  }

  @Benchmark
  public void buildFp32Index(Blackhole bh) {
    HnswConfig config = HnswConfig.withSeed(DEFAULT_DATA_SEED).withEfSearch(DEFAULT_EF_SEARCH);
    HnswIndex index = new HnswIndex(DEFAULT_DIMENSION, DEFAULT_METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      index.insert(i, rawDataset[i]);
    }
    bh.consume(index);
  }

  @Benchmark
  public void buildPureSq8Index(Blackhole bh) {
    HnswConfig config = HnswConfig.withSeed(DEFAULT_DATA_SEED).withEfSearch(DEFAULT_EF_SEARCH);
    QuantizedHnswIndex index =
        new QuantizedHnswIndex(DEFAULT_DIMENSION, DEFAULT_METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      index.insert(i, rawDataset[i]);
    }
    bh.consume(index);
  }

  @Benchmark
  public void searchFp32Index(Blackhole bh) {
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      bh.consume(jmhFp32Index.searchKnn(rawQueries[q], DEFAULT_K, DEFAULT_EF_SEARCH));
    }
  }

  @Benchmark
  public void searchPureSq8Index(Blackhole bh) {
    for (int q = 0; q < DEFAULT_NUM_QUERIES; q++) {
      bh.consume(jmhPureSq8Index.searchKnn(rawQueries[q], DEFAULT_K, DEFAULT_EF_SEARCH));
    }
  }

  /** Result record capturing full multi-dimensional evaluation of a scale step. */
  public record StrategyEvaluationResult(
      int vectorCount,
      int dimension,
      int k,
      int efSearch,
      int numQueries,
      // Build Performance
      long fp32BuildTimeMs,
      double fp32BuildThroughput,
      long pureSq8BuildTimeMs,
      double pureSq8BuildThroughput,
      double buildTimeRatio,
      // Graph Topology & Divergence (Pure SQ8 vs FP32 Reference)
      double layer0EdgeJaccard,
      long fp32TotalEdges,
      long pureSq8TotalEdges,
      long sharedEdgesLayer0,
      int pureSq8MaxDegreeL0,
      int pureSq8IsolatedL0,
      int pureSq8ComponentsL0,
      boolean pureSq8TopologyHealthy,
      // Search Recall (against exact FP32 Flat Ground Truth Oracle)
      double fp32Recall,
      double hybridSq8Recall,
      double pureSq8Recall,
      // Scientific Attribution of Recall Loss
      double totalRecallLoss,
      double quantizationDistanceLoss,
      double topologyDivergenceLoss,
      double hybridPureAgreement,
      // Search Latency & Throughput
      double fp32Qps,
      double fp32LatencyUs,
      double pureSq8Qps,
      double pureSq8LatencyUs,
      double searchSpeedup) {

    public String toBuildMarkdownRow() {
      return String.format(
          "| %,10d | %14d ms | %,17.1f | %15d ms | %,18.1f | %15.2fx |",
          vectorCount,
          fp32BuildTimeMs,
          fp32BuildThroughput,
          pureSq8BuildTimeMs,
          pureSq8BuildThroughput,
          buildTimeRatio);
    }

    public String toTopologyMarkdownRow() {
      return String.format(
          "| %,10d | %18.2f%% | %,16d | %,16d | %12d | %14d | %14d | %15b |",
          vectorCount,
          layer0EdgeJaccard * 100.0,
          fp32TotalEdges,
          pureSq8TotalEdges,
          pureSq8MaxDegreeL0,
          pureSq8IsolatedL0,
          pureSq8ComponentsL0,
          pureSq8TopologyHealthy);
    }

    public String toRecallMarkdownRow() {
      return String.format(
          "| %,10d | %15.2f%% | %17.2f%% | %15.2f%% | %15.2f%% | %17.2f%% | %18.2f%% | %19.2f%% |",
          vectorCount,
          fp32Recall * 100.0,
          hybridSq8Recall * 100.0,
          pureSq8Recall * 100.0,
          totalRecallLoss * 100.0,
          quantizationDistanceLoss * 100.0,
          topologyDivergenceLoss * 100.0,
          hybridPureAgreement * 100.0);
    }

    public String toSearchMarkdownRow() {
      return String.format(
          "| %,10d | %,14.1f | %15.2f | %,15.1f | %16.2f | %15.2fx |",
          vectorCount, fp32Qps, fp32LatencyUs, pureSq8Qps, pureSq8LatencyUs, searchSpeedup);
    }
  }

  /**
   * Executes a comprehensive evaluation of the three HNSW construction strategies at the specified
   * scale.
   */
  public static StrategyEvaluationResult evaluate(
      int n, int dimension, int k, int efSearch, int numQueries, long dataSeed, long querySeed) {

    // 1. Generate synthetic dataset and queries
    Random dataRng = new Random(dataSeed);
    float[][] dataset = new float[n][dimension];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dimension; d++) {
        dataset[i][d] = dataRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    Random queryRng = new Random(querySeed);
    float[][] queries = new float[numQueries][dimension];
    for (int q = 0; q < numQueries; q++) {
      for (int d = 0; d < dimension; d++) {
        queries[q][d] = queryRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    System.out.printf("  [N=%,d] 1/4 Building FP32 Flat Oracle & FP32 HNSW (efC=200)...\n", n);
    System.out.flush();

    // 2. Build Ground Truth Oracle (FP32 FlatIndex)
    FlatIndex oracle = new FlatIndex(dimension, DEFAULT_METRIC, n, true);
    for (int i = 0; i < n; i++) {
      oracle.insert(i, dataset[i]);
    }

    // 3. Build FP32 HNSW (Reference Oracle Graph)
    HnswConfig config = HnswConfig.withSeed(dataSeed).withEfSearch(efSearch);
    long startFp32Build = System.nanoTime();
    HnswIndex fp32Index = new HnswIndex(dimension, DEFAULT_METRIC, config, n, true);
    for (int i = 0; i < n; i++) {
      fp32Index.insert(i, dataset[i]);
    }
    long fp32ElapsedNs = System.nanoTime() - startFp32Build;
    long fp32BuildTimeMs = fp32ElapsedNs / 1_000_000L;
    double fp32BuildThroughput = (double) n / (fp32ElapsedNs / 1_000_000_000.0);

    // 4. Construct Hybrid SQ8 HNSW (Strategy A: Reuses FP32 graph, quantizes storage)
    QuantizedHnswIndex hybridIndex = QuantizedHnswIndex.fromFp32(fp32Index, true);

    System.out.printf(
        "  [N=%,d] 2/4 FP32 built (%d ms). Building Pure SQ8 HNSW (SIMD ADC)...\n",
        n, fp32BuildTimeMs);
    System.out.flush();

    // 5. Build Pure SQ8 HNSW (Strategy B: SQ8 SIMD ADC routing + construction)
    long startPureBuild = System.nanoTime();
    QuantizedHnswIndex pureSq8Index =
        new QuantizedHnswIndex(dimension, DEFAULT_METRIC, config, n, true);
    for (int i = 0; i < n; i++) {
      pureSq8Index.insert(i, dataset[i]);
    }
    long pureElapsedNs = System.nanoTime() - startPureBuild;
    long pureSq8BuildTimeMs = pureElapsedNs / 1_000_000L;
    double pureSq8BuildThroughput = (double) n / (pureElapsedNs / 1_000_000_000.0);
    double buildTimeRatio = (double) pureSq8BuildTimeMs / (double) Math.max(1L, fp32BuildTimeMs);

    System.out.printf(
        "  [N=%,d] 3/4 Pure SQ8 built (%d ms). Analyzing Topology Divergence...\n",
        n, pureSq8BuildTimeMs);
    System.out.flush();

    // 6. Analyze Graph Topology & Divergence (Pure SQ8 vs FP32 Reference)
    HnswGraph fp32Graph = fp32Index.graph();
    HnswGraph pureSq8Graph = pureSq8Index.graph();

    double totalJaccard = 0.0;
    long sharedEdgesL0 = 0;
    long fp32TotalEdges = 0;
    long pureSq8TotalEdges = 0;
    int pureMaxDegreeL0 = 0;
    int pureMaxDegreeUpper = 0;
    int pureIsolatedL0 = 0;

    for (int i = 0; i < n; i++) {
      HnswNode nodeFp32 = fp32Graph.getNode(i);
      HnswNode nodePure = pureSq8Graph.getNode(i);

      for (int l = 0; l <= nodeFp32.maxLevel(); l++) {
        fp32TotalEdges += nodeFp32.degree(l);
      }
      for (int l = 0; l <= nodePure.maxLevel(); l++) {
        int deg = nodePure.degree(l);
        pureSq8TotalEdges += deg;
        if (l == 0) {
          pureMaxDegreeL0 = Math.max(pureMaxDegreeL0, deg);
          if (deg == 0 && n > 1) {
            pureIsolatedL0++;
          }
        } else {
          pureMaxDegreeUpper = Math.max(pureMaxDegreeUpper, deg);
        }
      }

      int[] fp32L0 = nodeFp32.getNeighbors(0);
      int[] pureL0 = nodePure.getNeighbors(0);

      int intersection = 0;
      for (int a : fp32L0) {
        for (int b : pureL0) {
          if (a == b) {
            intersection++;
            break;
          }
        }
      }
      int union = fp32L0.length + pureL0.length - intersection;
      double nodeJaccard = (union == 0) ? 1.0 : (double) intersection / union;
      totalJaccard += nodeJaccard;
      sharedEdgesL0 += intersection;
    }

    double meanLayer0Jaccard = totalJaccard / n;

    // Check connected components on Layer 0 via BFS for Pure SQ8 graph
    int pureComponentsL0 = 0;
    if (n > 0) {
      boolean[] visited = new boolean[n];
      for (int i = 0; i < n; i++) {
        if (!visited[i]) {
          pureComponentsL0++;
          int[] queue = new int[n];
          int head = 0;
          int tail = 0;
          queue[tail++] = i;
          visited[i] = true;
          while (head < tail) {
            int curr = queue[head++];
            for (int neighbor : pureSq8Graph.getNode(curr).getNeighbors(0)) {
              if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue[tail++] = neighbor;
              }
            }
          }
        }
      }
    }

    boolean pureTopologyHealthy =
        pureComponentsL0 == 1
            && pureIsolatedL0 == 0
            && pureMaxDegreeL0 <= config.m0()
            && pureMaxDegreeUpper <= config.m();

    System.out.printf(
        "  [N=%,d] 4/4 Topology healthy=%b (Jaccard=%.2f%%). Evaluating Recall@10 & Latency...\n",
        n, pureTopologyHealthy, meanLayer0Jaccard * 100.0);
    System.out.flush();

    // 7. Measure Search Recall@10 against Ground Truth
    double totalFp32Recall = 0.0;
    double totalHybridRecall = 0.0;
    double totalPureRecall = 0.0;
    double totalMutualAgreement = 0.0;

    for (int q = 0; q < numQueries; q++) {
      float[] query = queries[q];
      List<SearchResult> gtResults = oracle.searchKnn(query, k);
      List<SearchResult> fp32Results = fp32Index.searchKnn(query, k, efSearch);
      List<SearchResult> hybridResults = hybridIndex.searchKnn(query, k, efSearch);
      List<SearchResult> pureResults = pureSq8Index.searchKnn(query, k, efSearch);

      Set<Long> gtIds = new HashSet<>(k);
      for (SearchResult r : gtResults) {
        gtIds.add(r.id());
      }

      Set<Long> hybridIds = new HashSet<>(k);
      for (SearchResult r : hybridResults) {
        hybridIds.add(r.id());
      }

      int fp32Matches = 0;
      for (SearchResult r : fp32Results) {
        if (gtIds.contains(r.id())) {
          fp32Matches++;
        }
      }

      int hybridMatches = 0;
      for (long id : hybridIds) {
        if (gtIds.contains(id)) {
          hybridMatches++;
        }
      }

      int pureMatches = 0;
      int mutualMatches = 0;
      for (SearchResult r : pureResults) {
        if (gtIds.contains(r.id())) {
          pureMatches++;
        }
        if (hybridIds.contains(r.id())) {
          mutualMatches++;
        }
      }

      totalFp32Recall += (double) fp32Matches / k;
      totalHybridRecall += (double) hybridMatches / k;
      totalPureRecall += (double) pureMatches / k;
      totalMutualAgreement += (double) mutualMatches / k;
    }

    double meanFp32Recall = totalFp32Recall / numQueries;
    double meanHybridRecall = totalHybridRecall / numQueries;
    double meanPureRecall = totalPureRecall / numQueries;
    double totalRecallLoss = meanFp32Recall - meanPureRecall;
    double quantizationDistanceLoss = meanFp32Recall - meanHybridRecall;
    double topologyDivergenceLoss = meanHybridRecall - meanPureRecall;
    double hybridPureAgreement = totalMutualAgreement / numQueries;

    // 8. Measure Search Latency & Throughput (QPS)
    // Warmup pass
    int warmupCount = Math.min(32, numQueries);
    for (int q = 0; q < warmupCount; q++) {
      fp32Index.searchKnn(queries[q], k, efSearch);
      pureSq8Index.searchKnn(queries[q], k, efSearch);
    }

    // Timed search: 5 passes over queries
    int searchPasses = 5;
    int totalSearchOps = numQueries * searchPasses;

    long startFp32Search = System.nanoTime();
    for (int p = 0; p < searchPasses; p++) {
      for (int q = 0; q < numQueries; q++) {
        fp32Index.searchKnn(queries[q], k, efSearch);
      }
    }
    long elapsedFp32SearchNs = System.nanoTime() - startFp32Search;
    double fp32Qps = (double) totalSearchOps / (elapsedFp32SearchNs / 1_000_000_000.0);
    double fp32LatencyUs = (double) elapsedFp32SearchNs / totalSearchOps / 1000.0;

    long startPureSearch = System.nanoTime();
    for (int p = 0; p < searchPasses; p++) {
      for (int q = 0; q < numQueries; q++) {
        pureSq8Index.searchKnn(queries[q], k, efSearch);
      }
    }
    long elapsedPureSearchNs = System.nanoTime() - startPureSearch;
    double pureSq8Qps = (double) totalSearchOps / (elapsedPureSearchNs / 1_000_000_000.0);
    double pureSq8LatencyUs = (double) elapsedPureSearchNs / totalSearchOps / 1000.0;
    double searchSpeedup = pureSq8Qps / Math.max(1e-9, fp32Qps);

    return new StrategyEvaluationResult(
        n,
        dimension,
        k,
        efSearch,
        numQueries,
        fp32BuildTimeMs,
        fp32BuildThroughput,
        pureSq8BuildTimeMs,
        pureSq8BuildThroughput,
        buildTimeRatio,
        meanLayer0Jaccard,
        fp32TotalEdges,
        pureSq8TotalEdges,
        sharedEdgesL0,
        pureMaxDegreeL0,
        pureIsolatedL0,
        pureComponentsL0,
        pureTopologyHealthy,
        meanFp32Recall,
        meanHybridRecall,
        meanPureRecall,
        totalRecallLoss,
        quantizationDistanceLoss,
        topologyDivergenceLoss,
        hybridPureAgreement,
        fp32Qps,
        fp32LatencyUs,
        pureSq8Qps,
        pureSq8LatencyUs,
        searchSpeedup);
  }

  /** Runs the evaluation across an array of dataset scales. */
  public static List<StrategyEvaluationResult> runSweep(int[] scales) {
    List<StrategyEvaluationResult> results = new ArrayList<>();
    for (int n : scales) {
      System.out.printf(
          "\n>>> Commencing evaluation for Scale N = %,d (D=%d, efC=200, efS=%d, Q=%d)...\n",
          n, DEFAULT_DIMENSION, DEFAULT_EF_SEARCH, DEFAULT_NUM_QUERIES);
      System.out.flush();
      StrategyEvaluationResult r =
          evaluate(
              n,
              DEFAULT_DIMENSION,
              DEFAULT_K,
              DEFAULT_EF_SEARCH,
              DEFAULT_NUM_QUERIES,
              DEFAULT_DATA_SEED,
              DEFAULT_QUERY_SEED);
      results.add(r);
      System.out.printf(
          ">>> Completed Scale N = %,d | FP32: %d ms | Pure SQ8: %d ms | L0 Jaccard: %.2f%% | Recall: FP32=%.2f%%, Hybrid=%.2f%%, Pure=%.2f%%\n\n",
          n,
          r.fp32BuildTimeMs(),
          r.pureSq8BuildTimeMs(),
          r.layer0EdgeJaccard() * 100.0,
          r.fp32Recall() * 100.0,
          r.hybridSq8Recall() * 100.0,
          r.pureSq8Recall() * 100.0);
      System.out.flush();
    }
    return results;
  }

  public static void printReportTables(List<StrategyEvaluationResult> results) {
    System.out.println(
        "====================================================================================================================");
    System.out.println(
        "                     PHASE 6C: HNSW GRAPH CONSTRUCTION STRATEGIES & TOPOLOGY STUDY                                  ");
    System.out.println(
        "====================================================================================================================");
    System.out.printf(
        "Configuration: D=%d, Metric=%s, k=%d, efSearch=%d, Queries=%d, DataSeed=%d, QuerySeed=%d\n\n",
        DEFAULT_DIMENSION,
        DEFAULT_METRIC,
        DEFAULT_K,
        DEFAULT_EF_SEARCH,
        DEFAULT_NUM_QUERIES,
        DEFAULT_DATA_SEED,
        DEFAULT_QUERY_SEED);

    System.out.println("### Table 1: Construction Performance & Build Throughput Comparison");
    System.out.println(
        "|          N | FP32 Build Time | FP32 Throughput (v/s) | Pure SQ8 Build Time | Pure SQ8 Throughput | Build Time Ratio |");
    System.out.println(
        "|-----------:|----------------:|----------------------:|--------------------:|--------------------:|-----------------:|");
    for (StrategyEvaluationResult r : results) {
      System.out.println(r.toBuildMarkdownRow());
    }
    System.out.println();

    System.out.println(
        "### Table 2: Graph Topological Divergence & Invariant Health (Pure SQ8 vs FP32 Reference)");
    System.out.println(
        "|          N | Layer-0 Jaccard Sim |  FP32 Total Edges | Pure SQ8 Edges | Max Deg L0 | Isolated L0 | Components L0 | Topology Health |");
    System.out.println(
        "|-----------:|--------------------:|------------------:|---------------:|-----------:|------------:|--------------:|----------------:|");
    for (StrategyEvaluationResult r : results) {
      System.out.println(r.toTopologyMarkdownRow());
    }
    System.out.println();

    System.out.println("### Table 3: Search Recall@10 Breakdown & Error Attribution");
    System.out.println(
        "|          N | FP32 HNSW Recall | Hybrid SQ8 Recall | Pure SQ8 Recall | Total Recall Loss | Quant Distance Loss | Topology Loss (Attrib) | Hybrid-Pure Agree |");
    System.out.println(
        "|-----------:|-----------------:|------------------:|----------------:|------------------:|--------------------:|-----------------------:|------------------:|");
    for (StrategyEvaluationResult r : results) {
      System.out.println(r.toRecallMarkdownRow());
    }
    System.out.println();

    System.out.println("### Table 4: Search Latency & Throughput Speedup");
    System.out.println(
        "|          N | FP32 Throughput | FP32 Latency (us) | Pure SQ8 Throughput | Pure SQ8 Latency | Search Speedup |");
    System.out.println(
        "|-----------:|----------------:|------------------:|--------------------:|-----------------:|---------------:|");
    for (StrategyEvaluationResult r : results) {
      System.out.println(r.toSearchMarkdownRow());
    }
    System.out.println(
        "====================================================================================================================\n");
  }

  public static void main(String[] args) throws RunnerException {
    if (args.length > 0 && "--jmh".equals(args[0])) {
      Options opt =
          new OptionsBuilder()
              .include(HnswConstructionStrategyBenchmark.class.getSimpleName())
              .forks(1)
              .build();
      new Runner(opt).run();
      return;
    }

    List<StrategyEvaluationResult> results = runSweep(DEFAULT_SCALES);
    printReportTables(results);
  }
}
