package com.nanovector.benchmark.allocation;

import com.nanovector.core.distance.DistanceCalculator;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.model.SearchResult;
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
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * JMH Allocation and Garbage Collection Profiling Benchmark (executed with {@code -prof gc}).
 *
 * <p>Empirically investigates and tests the <b>Heap Allocation Hypothesis</b>:
 *
 * <ul>
 *   <li><b>Raw Distance Baseline</b>: Primitive math kernel directly on contiguous buffers is
 *       hypothesized to produce strictly <b>0 B/op</b> heap allocation.
 *   <li><b>FlatIndex Search</b>: Distance scan loop allocates 0 B inside the $O(N)$ loop, but the
 *       bounded heap and returned Top-K results produce a small, $O(K)$ constant overhead invariant
 *       of dataset size $N$.
 *   <li><b>HNSW Graph Search</b>: While {@code EpochVisitedSet} avoids visited-set allocations, the
 *       graph search path allocates candidate nodes in the priority queues and wraps Top-K into
 *       {@link SearchResult} records, producing non-zero allocations scaling with {@code efSearch}.
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
public class SearchAllocationBenchmark {

  private static final int DIMENSION = 128;
  private static final DistanceMetric METRIC = DistanceMetric.EUCLIDEAN;
  private static final int K = 10;
  private static final int NUM_QUERIES = 128;
  private static final int QUERY_MASK = NUM_QUERIES - 1;

  @Param({"1000", "10000"})
  private int vectorCount;

  private FlatIndex flatIndex;
  private HnswIndex hnswIndex;
  private DistanceCalculator calculator;
  private float[] sampleVectorA;
  private float[] sampleVectorB;
  private float[][] queries;
  private int queryIndex;

  @Setup
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

    this.calculator = DistanceCalculator.create(METRIC, true);
    this.sampleVectorA = dataset[0];
    this.sampleVectorB = dataset[1];

    // Build FlatIndex (SIMD)
    this.flatIndex = new FlatIndex(DIMENSION, METRIC, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      this.flatIndex.insert(i, dataset[i]);
    }

    // Build HnswIndex (SIMD, efConstruction=200, M=16, M0=32)
    HnswConfig config = HnswConfig.withSeed(42L).withEfSearch(50);
    this.hnswIndex = new HnswIndex(DIMENSION, METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      this.hnswIndex.insert(i, dataset[i]);
    }

    this.queryIndex = 0;
  }

  /** Baseline: primitive distance evaluation kernel between two vectors. */
  @Benchmark
  public void baselineRawDistance(Blackhole bh) {
    float dist = calculator.distance(sampleVectorA, sampleVectorB);
    bh.consume(dist);
  }

  /** FlatIndex Top-K search: tests whether scan loop generates zero allocations. */
  @Benchmark
  public void searchFlat(Blackhole bh) {
    int idx = (queryIndex++) & QUERY_MASK;
    List<SearchResult> results = flatIndex.searchKnn(queries[idx], K);
    bh.consume(results);
  }

  /** HnswIndex search with efSearch=10: tight beam search candidate allocation. */
  @Benchmark
  public void searchHnswEf10(Blackhole bh) {
    int idx = (queryIndex++) & QUERY_MASK;
    List<SearchResult> results = hnswIndex.searchKnn(queries[idx], K, 10);
    bh.consume(results);
  }

  /** HnswIndex search with efSearch=50: medium beam search candidate allocation. */
  @Benchmark
  public void searchHnswEf50(Blackhole bh) {
    int idx = (queryIndex++) & QUERY_MASK;
    List<SearchResult> results = hnswIndex.searchKnn(queries[idx], K, 50);
    bh.consume(results);
  }

  /** HnswIndex search with efSearch=100: wide beam search candidate allocation. */
  @Benchmark
  public void searchHnswEf100(Blackhole bh) {
    int idx = (queryIndex++) & QUERY_MASK;
    List<SearchResult> results = hnswIndex.searchKnn(queries[idx], K, 100);
    bh.consume(results);
  }

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

  public static void main(String[] args) throws RunnerException {
    Options opt =
        new OptionsBuilder()
            .include(SearchAllocationBenchmark.class.getSimpleName())
            .addProfiler("gc")
            .forks(1)
            .build();
    new Runner(opt).run();
  }
}
