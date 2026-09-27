package com.nanovector.benchmark.scale;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.index.HnswIndex;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
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
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * JMH Benchmark and structural analyzer evaluating HNSW graph construction throughput, total build
 * latency, and graph topology health at scale ($N \in \{1\text{K}, 10\text{K}, 50\text{K},
 * 100\text{K}\}$ at dimension $D=128$).
 *
 * <p>Key experimental controls and measurements:
 *
 * <ul>
 *   <li>Evaluates end-to-end index build throughput (vectors/second) and average insertion latency
 *       ($\mu\text{s}$/vector) across a 100-fold dataset scale expansion.
 *   <li>Dataset generation ($D=128$, uniform synthetic vectors, seed 42) is decoupled from timed
 *       construction regions.
 *   <li>SIMD distance evaluation enabled during candidate exploration.
 *   <li>Graph topology analysis: multi-layer node distribution, degree distribution (min, mean,
 *       max) per layer, degree capping verification against $M$ and $M_0$, and isolation check
 *       (ensuring zero disconnected components at Layer 0).
 * </ul>
 */
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 0)
@Measurement(iterations = 1)
@Fork(
    value = 1,
    jvmArgsAppend = {"--add-modules", "jdk.incubator.vector"})
public class ScaleConstructionBenchmark {

  private static final int DIMENSION = 128;
  private static final DistanceMetric METRIC = DistanceMetric.EUCLIDEAN;

  @Param({"1000", "10000", "50000", "100000"})
  private int vectorCount;

  private float[][] dataset;

  @Setup
  public void setup() {
    Random rng = new Random(42L);
    this.dataset = new float[vectorCount][DIMENSION];
    for (int i = 0; i < vectorCount; i++) {
      for (int d = 0; d < DIMENSION; d++) {
        this.dataset[i][d] = rng.nextFloat() * 2.0f - 1.0f;
      }
    }
  }

  @Benchmark
  public HnswIndex buildHnswIndex() {
    HnswConfig config = HnswConfig.withSeed(42L).withEfSearch(50);
    HnswIndex index = new HnswIndex(DIMENSION, METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      index.insert(i, dataset[i]);
    }
    return index;
  }

  /** Record detailing graph structural properties and degree distribution for a single layer. */
  public record LayerStats(
      int layer,
      int nodeCount,
      double fractionOfTotal,
      int minDegree,
      double avgDegree,
      int maxDegree,
      int maxAllowedDegree,
      long totalEdges,
      boolean invariantPass) {

    public String toMarkdownRow() {
      return String.format(
          "| Layer %2d | %10d | %11.2f%% | %10d | %10.2f | %10d | %11d | %14b |",
          layer,
          nodeCount,
          fractionOfTotal * 100.0,
          minDegree,
          avgDegree,
          maxDegree,
          maxAllowedDegree,
          invariantPass);
    }
  }

  /** Record capturing end-to-end construction metrics and graph topology breakdown. */
  public record ConstructionReport(
      int vectorCount,
      int dimension,
      double rawPayloadMiB,
      long buildTimeMs,
      double throughputVecPerSec,
      double meanLatencyUsPerVec,
      int maxLevel,
      int isolatedNodesLayer0,
      int connectedComponentsLayer0,
      long totalEdgesAllLayers,
      List<LayerStats> layerStats) {

    public String toSummaryMarkdownRow() {
      return String.format(
          "| %7d | %13.2f | %15d | %22.1f | %16.2f | %9d | %11d | %14d | %11d |",
          vectorCount,
          rawPayloadMiB,
          buildTimeMs,
          throughputVecPerSec,
          meanLatencyUsPerVec,
          maxLevel,
          isolatedNodesLayer0,
          connectedComponentsLayer0,
          totalEdgesAllLayers);
    }
  }

  /**
   * Builds an HNSW index at the specified scale and derives detailed construction performance and
   * graph topology statistics.
   *
   * @param count number of vectors to index
   * @param dim vector dimension
   * @return detailed construction and topology report
   */
  public static ConstructionReport profileConstruction(int count, int dim) {
    Random rng = new Random(42L);
    float[][] data = new float[count][dim];
    for (int i = 0; i < count; i++) {
      for (int d = 0; d < dim; d++) {
        data[i][d] = rng.nextFloat() * 2.0f - 1.0f;
      }
    }

    HnswConfig config = HnswConfig.withSeed(42L).withEfSearch(50);
    long startNs = System.nanoTime();
    HnswIndex index = new HnswIndex(dim, METRIC, config, count, true);
    for (int i = 0; i < count; i++) {
      index.insert(i, data[i]);
    }
    long elapsedNs = System.nanoTime() - startNs;

    long buildTimeMs = elapsedNs / 1_000_000L;
    double throughput = (double) count / (elapsedNs / 1_000_000_000.0);
    double meanLatencyUs = (double) elapsedNs / count / 1000.0;
    double rawPayloadMiB = (double) count * dim * Float.BYTES / (1024.0 * 1024.0);

    HnswGraph graph = index.graph();
    int maxLevel = graph.maxLevel();

    List<LayerStats> layerStatsList = new ArrayList<>();
    long totalEdges = 0;

    for (int l = 0; l <= maxLevel; l++) {
      int nodeCount = 0;
      int minDeg = Integer.MAX_VALUE;
      int maxDeg = 0;
      long layerEdges = 0;
      int maxAllowed = (l == 0) ? config.m0() : config.m();

      for (int i = 0; i < count; i++) {
        HnswNode node = graph.getNode(i);
        if (node.maxLevel() >= l) {
          nodeCount++;
          int deg = node.degree(l);
          minDeg = Math.min(minDeg, deg);
          maxDeg = Math.max(maxDeg, deg);
          layerEdges += deg;
        }
      }

      double avgDeg = nodeCount > 0 ? (double) layerEdges / nodeCount : 0.0;
      boolean pass = maxDeg <= maxAllowed && (l > 0 || (minDeg > 0 || count <= 1));
      totalEdges += layerEdges;

      layerStatsList.add(
          new LayerStats(
              l,
              nodeCount,
              (double) nodeCount / count,
              minDeg == Integer.MAX_VALUE ? 0 : minDeg,
              avgDeg,
              maxDeg,
              maxAllowed,
              layerEdges,
              pass));
    }

    int isolatedNodesLayer0 = 0;
    for (int i = 0; i < count; i++) {
      if (graph.getNode(i).degree(0) == 0 && count > 1) {
        isolatedNodesLayer0++;
      }
    }

    // Connected components on Layer 0 via BFS
    int connectedComponentsLayer0 = 0;
    if (count > 0) {
      boolean[] visited = new boolean[count];
      for (int i = 0; i < count; i++) {
        if (!visited[i]) {
          connectedComponentsLayer0++;
          int[] queue = new int[count];
          int head = 0;
          int tail = 0;
          queue[tail++] = i;
          visited[i] = true;
          while (head < tail) {
            int curr = queue[head++];
            for (int neighbor : graph.getNode(curr).getNeighbors(0)) {
              if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue[tail++] = neighbor;
              }
            }
          }
        }
      }
    }

    return new ConstructionReport(
        count,
        dim,
        rawPayloadMiB,
        buildTimeMs,
        throughput,
        meanLatencyUs,
        maxLevel,
        isolatedNodesLayer0,
        connectedComponentsLayer0,
        totalEdges,
        layerStatsList);
  }

  /**
   * Executes a scale sweep measuring construction performance and printing detailed topology
   * breakdown across dataset sizes.
   *
   * @param scales array of dataset scales to profile
   */
  public static void runConstructionSweep(int[] scales) {
    System.out.println(
        "=============================================================================================================================");
    System.out.println(
        "                         NANOVECTOR HNSW GRAPH CONSTRUCTION & TOPOLOGY SWEEP                                                 ");
    System.out.println(
        "=============================================================================================================================");
    System.out.println(
        "| Scale N | Raw Size (MB) | Build Time (ms) | Throughput (vec/s) | Latency (us/vec) | Max Level | Isolated L0 | Components L0 | Total Edges |");
    System.out.println(
        "|:--------|--------------:|----------------:|-------------------:|-----------------:|----------:|------------:|--------------:|------------:|");

    List<ConstructionReport> reports = new ArrayList<>();
    for (int n : scales) {
      ConstructionReport report = profileConstruction(n, DIMENSION);
      reports.add(report);
      System.out.println(report.toSummaryMarkdownRow());
    }

    System.out.println(
        "=================================================================================================================\n");

    // Print detailed topology breakdown for the largest scale
    ConstructionReport largest = reports.get(reports.size() - 1);
    System.out.println(
        String.format(
            "TOPOLOGY BREAKDOWN AT SCALE N = %d (Dimension=%d, M=%d, M0=%d):",
            largest.vectorCount(),
            largest.dimension(),
            largest.layerStats().get(1).maxAllowedDegree(),
            largest.layerStats().get(0).maxAllowedDegree()));
    System.out.println(
        "| Layer    | Node Count | % of Total  | Min Degree | Avg Degree | Max Degree | Max Allowed | Invariant Pass |");
    System.out.println(
        "|:---------|-----------:|------------:|-----------:|-----------:|-----------:|------------:|---------------:|");
    for (LayerStats ls : largest.layerStats()) {
      System.out.println(ls.toMarkdownRow());
    }
    System.out.println(
        "=================================================================================================================");
  }

  // Package-private accessors for unit tests
  void initForTest(int count) {
    this.vectorCount = count;
    setup();
  }

  float[][] dataset() {
    return dataset;
  }

  public static void main(String[] args) throws RunnerException {
    if (args.length > 0 && "--topology-profile".equals(args[0])) {
      runConstructionSweep(new int[] {1000, 10000, 50000, 100000});
      return;
    }
    Options opt =
        new OptionsBuilder()
            .include(ScaleConstructionBenchmark.class.getSimpleName())
            .forks(1)
            .build();
    new Runner(opt).run();
  }
}
