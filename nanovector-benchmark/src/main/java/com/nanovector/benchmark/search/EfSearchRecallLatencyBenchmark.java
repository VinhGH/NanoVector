package com.nanovector.benchmark.search;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
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
 * JMH Benchmark exploring the HNSW Pareto Frontier: Recall@K vs Latency / Throughput (QPS) across
 * dynamic beam search widths ({@code efSearch} &isin; {10, 20, 50, 100, 200}).
 *
 * <p>Key experimental controls:
 *
 * <ul>
 *   <li>Evaluated on a 10,000-vector dataset of 128-dimensional vectors with 128 test queries.
 *   <li>Ground truth Top-10 oracle computed via exact {@link FlatIndex} scan.
 *   <li>Calculates empirical Recall@10 in trial setup and correlates directly with throughput.
 *   <li>Compares both Scalar and SIMD acceleration regimes along the Pareto curve.
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
public class EfSearchRecallLatencyBenchmark {

  private static final int DIMENSION = 128;
  private static final DistanceMetric METRIC = DistanceMetric.EUCLIDEAN;
  private static final int K = 10;
  private static final int NUM_QUERIES = 128;
  private static final int QUERY_MASK = NUM_QUERIES - 1;

  @Param({"10", "20", "50", "100", "200"})
  private int efSearch;

  @Param({"false", "true"})
  private boolean useSimd;

  private int vectorCount = 10000;
  private HnswIndex hnswIndex;
  private float[][] queries;
  private int queryIndex;
  private double measuredRecall;

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

    // 1. Compute exact Ground Truth Top-10 Oracle via FlatIndex
    FlatIndex oracle = new FlatIndex(DIMENSION, METRIC, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      oracle.insert(i, dataset[i]);
    }

    Set<Long>[] groundTruths = new Set[NUM_QUERIES];
    for (int q = 0; q < NUM_QUERIES; q++) {
      List<SearchResult> exactResults = oracle.searchKnn(queries[q], K);
      Set<Long> gtIds = new HashSet<>(K);
      for (SearchResult r : exactResults) {
        gtIds.add(r.id());
      }
      groundTruths[q] = gtIds;
    }

    // 2. Build HNSW index
    HnswConfig config = HnswConfig.withSeed(42L).withEfSearch(efSearch);
    this.hnswIndex = new HnswIndex(DIMENSION, METRIC, config, vectorCount, useSimd);
    for (int i = 0; i < vectorCount; i++) {
      this.hnswIndex.insert(i, dataset[i]);
    }

    // 3. Compute empirical Recall@10 for the current trial
    int totalHits = 0;
    for (int q = 0; q < NUM_QUERIES; q++) {
      List<SearchResult> hnswResults = hnswIndex.searchKnn(queries[q], K, efSearch);
      Set<Long> gt = groundTruths[q];
      for (SearchResult r : hnswResults) {
        if (gt.contains(r.id())) {
          totalHits++;
        }
      }
    }
    this.measuredRecall = (double) totalHits / (NUM_QUERIES * K);
    this.queryIndex = 0;

    System.out.printf(
        "[Pareto Frontier] N=%d | efSearch=%3d | SIMD=%5b | Recall@10=%6.2f%%%n",
        vectorCount, efSearch, useSimd, measuredRecall * 100.0);
  }

  @Benchmark
  public void searchHnsw(Blackhole bh) {
    int idx = (queryIndex++) & QUERY_MASK;
    List<SearchResult> results = hnswIndex.searchKnn(queries[idx], K, efSearch);
    bh.consume(results);
  }

  public double getMeasuredRecall() {
    return measuredRecall;
  }

  void initForTest(int count, int ef, boolean simd) {
    this.vectorCount = count;
    this.efSearch = ef;
    this.useSimd = simd;
    setup();
  }

  HnswIndex hnswIndex() {
    return hnswIndex;
  }

  public static void main(String[] args) throws RunnerException {
    Options opt =
        new OptionsBuilder()
            .include(EfSearchRecallLatencyBenchmark.class.getSimpleName())
            .forks(1)
            .build();
    new Runner(opt).run();
  }
}
