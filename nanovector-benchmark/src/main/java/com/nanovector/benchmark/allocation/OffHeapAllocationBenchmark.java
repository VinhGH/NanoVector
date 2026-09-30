package com.nanovector.benchmark.allocation;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.QuantizedHnswIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.offheap.OffHeapQuantizedEuclideanDistance;
import com.nanovector.core.offheap.OffHeapQuantizedHnswIndex;
import com.nanovector.core.quantization.QuantizedEuclideanDistance;
import java.lang.foreign.MemorySegment;
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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.profile.GCProfiler;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * JMH Allocation and Garbage Collection Profiling Benchmark for Off-Heap vs On-Heap Vector Search
 * (executed with {@code -prof gc}).
 *
 * <p>Empirically investigates GC churn, allocation rate (MB/s), and normalized allocations (B/op)
 * across:
 *
 * <ul>
 *   <li><b>Distance Hot Path</b>: {@code byte[]} on-heap vs {@link MemorySegment} off-heap.
 *   <li><b>HNSW Search Traversal</b>: On-heap FP32 vs On-heap SQ8 vs Off-heap SQ8.
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
public class OffHeapAllocationBenchmark {

  private static final int DIMENSION = 128;
  private static final DistanceMetric METRIC = DistanceMetric.EUCLIDEAN;
  private static final int K = 10;
  private static final int NUM_QUERIES = 128;
  private static final int QUERY_MASK = NUM_QUERIES - 1;

  @Param({"1000", "10000"})
  private int vectorCount;

  private HnswIndex onHeapFp32Index;
  private QuantizedHnswIndex onHeapSq8Index;
  private OffHeapQuantizedHnswIndex offHeapSq8Index;

  private QuantizedEuclideanDistance onHeapCalculator;
  private OffHeapQuantizedEuclideanDistance offHeapCalculator;

  private byte[] sampleOnHeapBuffer;
  private float sampleMin;
  private float sampleScale;

  private MemorySegment sampleOffHeapSegment;
  private long sampleOffHeapOffset;

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

    HnswConfig config = HnswConfig.withSeed(42L).withEfSearch(50);

    // 1. Build On-Heap FP32 HNSW
    this.onHeapFp32Index = new HnswIndex(DIMENSION, METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      this.onHeapFp32Index.insert(i, dataset[i]);
    }

    // 2. Build On-Heap SQ8 HNSW
    this.onHeapSq8Index = new QuantizedHnswIndex(DIMENSION, METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      this.onHeapSq8Index.insert(i, dataset[i]);
    }

    // 3. Build Off-Heap SQ8 HNSW
    this.offHeapSq8Index =
        new OffHeapQuantizedHnswIndex(DIMENSION, METRIC, config, vectorCount, true);
    for (int i = 0; i < vectorCount; i++) {
      this.offHeapSq8Index.insert(i, dataset[i]);
    }

    // Calculators
    this.onHeapCalculator = QuantizedEuclideanDistance.create(true);
    this.offHeapCalculator = OffHeapQuantizedEuclideanDistance.create(true);

    // Distance scan sample handles
    this.sampleOnHeapBuffer = onHeapSq8Index.storage().vectorBuffer();
    this.sampleMin = onHeapSq8Index.storage().getMin(0);
    this.sampleScale = onHeapSq8Index.storage().getScale(0);

    this.sampleOffHeapSegment = offHeapSq8Index.storage().vectorSegment();
    this.sampleOffHeapOffset = offHeapSq8Index.storage().getVectorOffset(0);

    this.queryIndex = 0;
  }

  @TearDown
  public void tearDown() {
    if (offHeapSq8Index != null) {
      offHeapSq8Index.close();
    }
  }

  // ── Hot-path Distance Scans ─────────────────────────────────────────

  @Benchmark
  public void onHeapDistance_SingleScan(Blackhole bh) {
    float[] query = queries[queryIndex++ & QUERY_MASK];
    float dist = onHeapCalculator.distance(sampleOnHeapBuffer, 0, sampleMin, sampleScale, query);
    bh.consume(dist);
  }

  @Benchmark
  public void offHeapDistance_SingleScan(Blackhole bh) {
    float[] query = queries[queryIndex++ & QUERY_MASK];
    float dist =
        offHeapCalculator.distance(
            sampleOffHeapSegment, sampleOffHeapOffset, sampleMin, sampleScale, query);
    bh.consume(dist);
  }

  // ── Full k-NN Search Traversal ──────────────────────────────────────

  @Benchmark
  public void onHeapFp32Hnsw_Search(Blackhole bh) {
    float[] query = queries[queryIndex++ & QUERY_MASK];
    List<SearchResult> results = onHeapFp32Index.searchKnn(query, K);
    bh.consume(results);
  }

  @Benchmark
  public void onHeapQuantizedHnsw_Search(Blackhole bh) {
    float[] query = queries[queryIndex++ & QUERY_MASK];
    List<SearchResult> results = onHeapSq8Index.searchKnn(query, K);
    bh.consume(results);
  }

  @Benchmark
  public void offHeapQuantizedHnsw_Search(Blackhole bh) {
    float[] query = queries[queryIndex++ & QUERY_MASK];
    List<SearchResult> results = offHeapSq8Index.searchKnn(query, K);
    bh.consume(results);
  }

  void initForTest(int count) {
    this.vectorCount = count;
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
    Options opt =
        new OptionsBuilder()
            .include(OffHeapAllocationBenchmark.class.getSimpleName())
            .addProfiler(GCProfiler.class)
            .build();
    new Runner(opt).run();
  }
}
