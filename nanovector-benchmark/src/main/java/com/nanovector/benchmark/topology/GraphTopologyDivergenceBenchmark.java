package com.nanovector.benchmark.topology;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.model.SearchResult;
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
 * Empirical analysis and JMH benchmark evaluating HNSW Graph Topology Divergence and Index
 * Construction Throughput between Scalar and SIMD distance engines.
 *
 * <p>Scientific investigation:
 *
 * <ul>
 *   <li><b>Topology Invariance vs Floating-Point Drift</b>: Tests whether associativity differences
 *       in SIMD dot-product reductions cause neighbor routing divergence during insertion.
 *   <li><b>Quantitative Edge Jaccard Similarity</b>: Measures the exact proportion of identical
 *       edges and node-level topology consistency across layers.
 *   <li><b>Build Time Acceleration</b>: Measures the throughput (ops/s or build time) of indexing
 *       vectors with Scalar vs SIMD distance calculations.
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
public class GraphTopologyDivergenceBenchmark {

  private static final int DIMENSION = 128;
  private static final DistanceMetric METRIC = DistanceMetric.EUCLIDEAN;

  @Param({"1000"})
  private int vectorCount;

  private float[][] rawDataset;

  @Setup
  public void setup() {
    Random rng = new Random(42L);
    this.rawDataset = new float[vectorCount][DIMENSION];
    for (int i = 0; i < vectorCount; i++) {
      for (int d = 0; d < DIMENSION; d++) {
        this.rawDataset[i][d] = rng.nextFloat() * 2.0f - 1.0f;
      }
    }
  }

  @Benchmark
  public void buildScalarIndex(Blackhole bh) {
    HnswConfig config = HnswConfig.withSeed(42L).withEfSearch(50);
    HnswIndex index = new HnswIndex(DIMENSION, METRIC, config, vectorCount, false);
    for (int i = 0; i < vectorCount; i++) {
      index.insert(i, rawDataset[i]);
    }
    bh.consume(index);
  }

  @Benchmark
  public void buildSimdIndex(Blackhole bh) {
    HnswConfig config = HnswConfig.withSeed(42L).withEfSearch(50);
    HnswIndex index = new HnswIndex(DIMENSION, METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      index.insert(i, rawDataset[i]);
    }
    bh.consume(index);
  }

  public record DivergenceReport(
      int vectorCount,
      int dimension,
      int entryPointScalar,
      int entryPointSimd,
      int maxLevelScalar,
      int maxLevelSimd,
      long totalEdgesScalar,
      long totalEdgesSimd,
      long sharedEdges,
      double edgeJaccardSimilarity,
      double identicalNodeRatio,
      double searchResultOverlapRatio) {}

  /**
   * Performs an exhaustive empirical comparison between a Scalar-constructed graph and a
   * SIMD-constructed graph using the same input vectors and random seed.
   */
  public static DivergenceReport analyzeDivergence(int n, int dim, long seed) {
    Random dataRng = new Random(seed);
    float[][] dataset = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        dataset[i][d] = dataRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    HnswConfig config = HnswConfig.withSeed(seed).withEfSearch(50);

    // Build Scalar graph
    HnswIndex scalarIndex = new HnswIndex(dim, METRIC, config, n, false);
    for (int i = 0; i < n; i++) {
      scalarIndex.insert(i, dataset[i]);
    }

    // Build SIMD graph
    HnswIndex simdIndex = new HnswIndex(dim, METRIC, config, n, true);
    for (int i = 0; i < n; i++) {
      simdIndex.insert(i, dataset[i]);
    }

    HnswGraph scalarGraph = scalarIndex.graph();
    HnswGraph simdGraph = simdIndex.graph();

    long totalEdgesScalar = 0;
    long totalEdgesSimd = 0;
    long sharedEdges = 0;
    long totalUnionEdges = 0;
    int identicalNodes = 0;

    for (int i = 0; i < n; i++) {
      HnswNode nodeScalar = scalarGraph.getNode(i);
      HnswNode nodeSimd = simdGraph.getNode(i);

      boolean nodeMatches = (nodeScalar.maxLevel() == nodeSimd.maxLevel());
      int maxL = Math.max(nodeScalar.maxLevel(), nodeSimd.maxLevel());

      for (int l = 0; l <= maxL; l++) {
        int[] sNeighbors = l <= nodeScalar.maxLevel() ? nodeScalar.getNeighbors(l) : new int[0];
        int[] vNeighbors = l <= nodeSimd.maxLevel() ? nodeSimd.getNeighbors(l) : new int[0];

        Set<Integer> sSet = toSet(sNeighbors);
        Set<Integer> vSet = toSet(vNeighbors);

        totalEdgesScalar += sSet.size();
        totalEdgesSimd += vSet.size();

        Set<Integer> intersection = new HashSet<>(sSet);
        intersection.retainAll(vSet);
        sharedEdges += intersection.size();

        Set<Integer> union = new HashSet<>(sSet);
        union.addAll(vSet);
        totalUnionEdges += union.size();

        if (!sSet.equals(vSet)) {
          nodeMatches = false;
        }
      }

      if (nodeMatches) {
        identicalNodes++;
      }
    }

    double edgeJaccard = totalUnionEdges == 0 ? 1.0 : (double) sharedEdges / totalUnionEdges;
    double identicalNodeRatio = (double) identicalNodes / n;

    // Evaluate Search Result Overlap on 100 queries
    Random queryRng = new Random(seed + 100);
    int numQueries = 100;
    int k = 10;
    double totalOverlap = 0;
    for (int q = 0; q < numQueries; q++) {
      float[] query = new float[dim];
      for (int d = 0; d < dim; d++) {
        query[d] = queryRng.nextFloat() * 2.0f - 1.0f;
      }
      List<SearchResult> sRes = scalarIndex.searchKnn(query, k);
      List<SearchResult> vRes = simdIndex.searchKnn(query, k);

      Set<Long> sIds = new HashSet<>();
      for (SearchResult r : sRes) {
        sIds.add(r.id());
      }
      Set<Long> vIds = new HashSet<>();
      for (SearchResult r : vRes) {
        vIds.add(r.id());
      }

      Set<Long> intersect = new HashSet<>(sIds);
      intersect.retainAll(vIds);
      totalOverlap += (double) intersect.size() / k;
    }
    double avgSearchOverlap = totalOverlap / numQueries;

    return new DivergenceReport(
        n,
        dim,
        scalarGraph.entryPointId(),
        simdGraph.entryPointId(),
        scalarGraph.maxLevel(),
        simdGraph.maxLevel(),
        totalEdgesScalar,
        totalEdgesSimd,
        sharedEdges,
        edgeJaccard,
        identicalNodeRatio,
        avgSearchOverlap);
  }

  private static Set<Integer> toSet(int[] arr) {
    Set<Integer> set = new HashSet<>(arr.length);
    for (int val : arr) {
      set.add(val);
    }
    return set;
  }

  public static void main(String[] args) throws RunnerException {
    System.out.println("Running Topology Divergence Analysis...");
    DivergenceReport report = analyzeDivergence(1000, 128, 42L);
    System.out.printf(
        "--- Topology Divergence Report (N=%d, D=%d) ---%n",
        report.vectorCount(), report.dimension());
    System.out.printf(
        "Entry Point: Scalar=%d, SIMD=%d%n", report.entryPointScalar(), report.entryPointSimd());
    System.out.printf(
        "Max Level:   Scalar=%d, SIMD=%d%n", report.maxLevelScalar(), report.maxLevelSimd());
    System.out.printf(
        "Total Edges: Scalar=%d, SIMD=%d%n", report.totalEdgesScalar(), report.totalEdgesSimd());
    System.out.printf(
        "Shared Edges: %d (Jaccard Similarity: %.4f%%)%n",
        report.sharedEdges(), report.edgeJaccardSimilarity() * 100.0);
    System.out.printf("Identical Nodes Ratio: %.2f%%%n", report.identicalNodeRatio() * 100.0);
    System.out.printf(
        "Search Top-10 Result Overlap: %.2f%%%n", report.searchResultOverlapRatio() * 100.0);

    Options opt =
        new OptionsBuilder()
            .include(GraphTopologyDivergenceBenchmark.class.getSimpleName())
            .forks(1)
            .build();
    new Runner(opt).run();
  }
}
