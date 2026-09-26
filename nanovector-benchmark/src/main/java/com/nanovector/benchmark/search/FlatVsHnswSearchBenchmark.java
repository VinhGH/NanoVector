package com.nanovector.benchmark.search;

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
 * JMH Benchmark evaluating end-to-end vector search throughput and latency across index
 * architectures (Flat vs HNSW) and distance engines (Scalar vs SIMD).
 *
 * <p>Key experimental controls:
 *
 * <ul>
 *   <li>Same dataset ($D=128$, uniform synthetic vectors, seed 42) for both indexes.
 *   <li>Same query set (128 random query vectors, seed 12345) cycled through per search.
 *   <li>Identical HNSW configuration ($M=16, M_0=32, efConstruction=200, efSearch=50, seed=42$).
 *   <li>Indexed dataset and graph structures constructed strictly in {@link #setup()} outside timed
 *       methods.
 *   <li>Evaluates Amdahl's Law by comparing raw kernel speedup propagation to end-to-end search.
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
public class FlatVsHnswSearchBenchmark {

  private static final int DIMENSION = 128;
  private static final DistanceMetric METRIC = DistanceMetric.EUCLIDEAN;
  private static final int K = 10;
  private static final int EF_SEARCH = 50;
  private static final int NUM_QUERIES = 128;
  private static final int QUERY_MASK = NUM_QUERIES - 1;

  @Param({"1000", "10000"})
  private int vectorCount;

  @Param({"false", "true"})
  private boolean useSimd;

  private FlatIndex flatIndex;
  private HnswIndex hnswIndex;
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

    this.flatIndex = new FlatIndex(DIMENSION, METRIC, vectorCount, useSimd);
    for (int i = 0; i < vectorCount; i++) {
      this.flatIndex.insert(i, dataset[i]);
    }

    HnswConfig config = HnswConfig.withSeed(42L).withEfSearch(EF_SEARCH);
    this.hnswIndex = new HnswIndex(DIMENSION, METRIC, config, vectorCount, useSimd);
    for (int i = 0; i < vectorCount; i++) {
      this.hnswIndex.insert(i, dataset[i]);
    }

    this.queryIndex = 0;

    // Sanity verification
    List<SearchResult> flatRes = this.flatIndex.searchKnn(this.queries[0], K);
    List<SearchResult> hnswRes = this.hnswIndex.searchKnn(this.queries[0], K);
    if (flatRes.size() != K || hnswRes.size() != K) {
      throw new IllegalStateException(
          String.format(
              "Search sanity check failed: flatCount=%d, hnswCount=%d, expected=%d",
              flatRes.size(), hnswRes.size(), K));
    }
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
    List<SearchResult> results = hnswIndex.searchKnn(queries[idx], K);
    bh.consume(results);
  }

  // Package-private accessors for unit test verification
  void initForTest(int count, boolean simd) {
    this.vectorCount = count;
    this.useSimd = simd;
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
            .include(FlatVsHnswSearchBenchmark.class.getSimpleName())
            .forks(1)
            .build();
    new Runner(opt).run();
  }
}
