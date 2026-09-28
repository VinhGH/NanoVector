package com.nanovector.benchmark.quantization;

import com.nanovector.benchmark.memory.MemoryFootprintProfiler;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.QuantizedFlatIndex;
import com.nanovector.core.model.SearchResult;
import java.util.ArrayList;
import java.util.Arrays;
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

/**
 * Benchmark evaluating Memory Footprint, Throughput, and Latency distribution (P50, P95, P99)
 * comparing full-precision FP32 FlatIndex against 8-bit Scalar Quantization (SQ8)
 * QuantizedFlatIndex across dataset scales ($N \in \{1\text{K}, 10\text{K}, 50\text{K},
 * 100\text{K}\}$ at dimension $D=128$).
 *
 * <p>Examines the central systems question of Phase 6B: Does reducing vector data by $\approx
 * 4\times$ translate to measurable search speedup, or does the CPU widening overhead (byte to
 * float) offset the memory bandwidth reduction?
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 3, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(
    value = 1,
    jvmArgsAppend = {"--add-modules", "jdk.incubator.vector"})
public class FlatVsQuantizedFlatBenchmark {

  public static final int DIMENSION = 128;
  public static final DistanceMetric METRIC = DistanceMetric.EUCLIDEAN;
  public static final int K = 10;
  public static final int NUM_QUERIES = 128;
  public static final int QUERY_MASK = NUM_QUERIES - 1;
  public static final int[] DEFAULT_SCALES = {1_000, 10_000, 50_000, 100_000};

  @Param({"1000", "10000", "50000", "100000"})
  private int vectorCount;

  private FlatIndex fp32Scalar;
  private FlatIndex fp32Simd;
  private QuantizedFlatIndex sq8Scalar;
  private QuantizedFlatIndex sq8Simd;
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

    this.fp32Scalar = new FlatIndex(DIMENSION, METRIC, vectorCount, false);
    this.fp32Simd = new FlatIndex(DIMENSION, METRIC, vectorCount, true);
    this.sq8Scalar = new QuantizedFlatIndex(DIMENSION, vectorCount, false);
    this.sq8Simd = new QuantizedFlatIndex(DIMENSION, vectorCount, true);

    for (int i = 0; i < vectorCount; i++) {
      this.fp32Scalar.insert(i, dataset[i]);
      this.fp32Simd.insert(i, dataset[i]);
      this.sq8Scalar.insert(i, dataset[i]);
      this.sq8Simd.insert(i, dataset[i]);
    }

    this.queryIndex = 0;
  }

  @Benchmark
  public void searchFp32Scalar(Blackhole bh) {
    float[] q = queries[queryIndex++ & QUERY_MASK];
    bh.consume(fp32Scalar.searchKnn(q, K));
  }

  @Benchmark
  public void searchFp32Simd(Blackhole bh) {
    float[] q = queries[queryIndex++ & QUERY_MASK];
    bh.consume(fp32Simd.searchKnn(q, K));
  }

  @Benchmark
  public void searchSq8Scalar(Blackhole bh) {
    float[] q = queries[queryIndex++ & QUERY_MASK];
    bh.consume(sq8Scalar.searchKnn(q, K));
  }

  @Benchmark
  public void searchSq8Simd(Blackhole bh) {
    float[] q = queries[queryIndex++ & QUERY_MASK];
    bh.consume(sq8Simd.searchKnn(q, K));
  }

  public enum Candidate {
    FP32_SCALAR("FP32 Flat (Scalar)"),
    FP32_SIMD("FP32 Flat (SIMD)"),
    SQ8_SCALAR("SQ8 Flat (Scalar ADC)"),
    SQ8_SIMD("SQ8 Flat (SIMD ADC)");

    private final String label;

    Candidate(String label) {
      this.label = label;
    }

    public String label() {
      return label;
    }
  }

  public record LatencyProfile(
      Candidate candidate,
      int vectorCount,
      double throughputQps,
      double meanLatencyUs,
      double p50LatencyUs,
      double p95LatencyUs,
      double p99LatencyUs,
      double speedupVsFp32) {

    public String toMarkdownRow() {
      return String.format(
          "| %,10d | %-24s | %,16.1f | %,15.2f | %,14.2f | %,14.2f | %,14.2f | %11.2fx |",
          vectorCount,
          candidate.label(),
          throughputQps,
          meanLatencyUs,
          p50LatencyUs,
          p95LatencyUs,
          p99LatencyUs,
          speedupVsFp32);
    }
  }

  public record MemoryComparisonRow(
      int n,
      double fp32RawMb,
      double sq8RawMb,
      double fp32VectorStorageBytesPerVec,
      double sq8VectorStorageBytesPerVec,
      double vectorStorageReductionRatio,
      double fp32TotalStructuralMb,
      double sq8TotalStructuralMb,
      double totalStructuralReductionRatio,
      double fp32MeasuredHeapDeltaMb,
      double sq8MeasuredHeapDeltaMb,
      double measuredHeapDeltaReductionRatio) {

    public String toMarkdownRow() {
      return String.format(
          "| %,10d | %11.2f | %10.2f | %13.0f B | %12.0f B | %15.2fx | %14.2f | %13.2f | %15.2fx | %13.2f | %12.2f | %16.2fx |",
          n,
          fp32RawMb,
          sq8RawMb,
          fp32VectorStorageBytesPerVec,
          sq8VectorStorageBytesPerVec,
          vectorStorageReductionRatio,
          fp32TotalStructuralMb,
          sq8TotalStructuralMb,
          totalStructuralReductionRatio,
          fp32MeasuredHeapDeltaMb,
          sq8MeasuredHeapDeltaMb,
          measuredHeapDeltaReductionRatio);
    }
  }

  public LatencyProfile profile(
      Candidate candidate, int numWarmup, int numSamples, double baselineLatencyUs) {
    for (int i = 0; i < numWarmup; i++) {
      float[] q = queries[i & QUERY_MASK];
      List<SearchResult> r =
          switch (candidate) {
            case FP32_SCALAR -> fp32Scalar.searchKnn(q, K);
            case FP32_SIMD -> fp32Simd.searchKnn(q, K);
            case SQ8_SCALAR -> sq8Scalar.searchKnn(q, K);
            case SQ8_SIMD -> sq8Simd.searchKnn(q, K);
          };
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
          switch (candidate) {
            case FP32_SCALAR -> fp32Scalar.searchKnn(q, K);
            case FP32_SIMD -> fp32Simd.searchKnn(q, K);
            case SQ8_SCALAR -> sq8Scalar.searchKnn(q, K);
            case SQ8_SIMD -> sq8Simd.searchKnn(q, K);
          };
      long end = System.nanoTime();
      latenciesNs[i] = end - start;
      if (r.isEmpty()) {
        throw new IllegalStateException("Empty search result during measurement");
      }
    }
    long totalEnd = System.nanoTime();

    Arrays.sort(latenciesNs);
    double p50 = latenciesNs[(int) (numSamples * 0.50)] / 1000.0;
    double p95 = latenciesNs[(int) (numSamples * 0.95)] / 1000.0;
    double p99 = latenciesNs[(int) (numSamples * 0.99)] / 1000.0;

    long sumNs = 0;
    for (long l : latenciesNs) {
      sumNs += l;
    }
    double meanUs = (double) sumNs / numSamples / 1000.0;
    double totalSeconds = (totalEnd - totalStart) / 1_000_000_000.0;
    double throughput = numSamples / totalSeconds;
    double speedup = baselineLatencyUs > 0 ? baselineLatencyUs / meanUs : 1.0;

    return new LatencyProfile(candidate, vectorCount, throughput, meanUs, p50, p95, p99, speedup);
  }

  public void initForTest(int count) {
    this.vectorCount = count;
    setup();
  }

  public FlatIndex fp32Scalar() {
    return fp32Scalar;
  }

  public FlatIndex fp32Simd() {
    return fp32Simd;
  }

  public QuantizedFlatIndex sq8Scalar() {
    return sq8Scalar;
  }

  public QuantizedFlatIndex sq8Simd() {
    return sq8Simd;
  }

  public static MemoryComparisonRow evaluateMemory(int n) {
    MemoryFootprintProfiler.MemoryReport fp32Rep =
        MemoryFootprintProfiler.profileFlat(n, DIMENSION);
    MemoryFootprintProfiler.MemoryReport sq8Rep =
        MemoryFootprintProfiler.profileQuantizedFlat(n, DIMENSION);

    double fp32RawMb = fp32Rep.rawVectorPayloadBytes() / (1024.0 * 1024.0);
    double sq8RawPayloadBytes = (double) n * (DIMENSION + 8);
    double sq8RawMb = sq8RawPayloadBytes / (1024.0 * 1024.0);

    double fp32VecStorageBytesPerVec = (double) (DIMENSION * Float.BYTES); // 512 B
    double sq8VecStorageBytesPerVec = (double) (DIMENSION * Byte.BYTES + 8); // 136 B
    double vecStorageReduction = fp32VecStorageBytesPerVec / sq8VecStorageBytesPerVec; // 3.76x

    double fp32TotalStructMb = fp32Rep.totalStructuralBytes() / (1024.0 * 1024.0);
    double sq8TotalStructMb = sq8Rep.totalStructuralBytes() / (1024.0 * 1024.0);
    double totalStructReduction = fp32TotalStructMb / sq8TotalStructMb;

    double fp32HeapDeltaMb = fp32Rep.measuredHeapDeltaBytes() / (1024.0 * 1024.0);
    double sq8HeapDeltaMb = sq8Rep.measuredHeapDeltaBytes() / (1024.0 * 1024.0);
    double heapDeltaReduction = sq8HeapDeltaMb > 0 ? fp32HeapDeltaMb / sq8HeapDeltaMb : 0.0;

    return new MemoryComparisonRow(
        n,
        fp32RawMb,
        sq8RawMb,
        fp32VecStorageBytesPerVec,
        sq8VecStorageBytesPerVec,
        vecStorageReduction,
        fp32TotalStructMb,
        sq8TotalStructMb,
        totalStructReduction,
        fp32HeapDeltaMb,
        sq8HeapDeltaMb,
        heapDeltaReduction);
  }

  public static void runSweep(int[] scales) {
    System.out.println(
        "==========================================================================================================================");
    System.out.println(
        "                 PHASE 6B COMMIT 4: FP32 FLAT vs SQ8 FLAT MEMORY & PERFORMANCE STUDY (D=128, k=10)                       ");
    System.out.println(
        "==========================================================================================================================\n");

    System.out.println("### 1. Memory Footprint: Structural vs Measured Heap Delta");
    System.out.println(
        "|          N | FP32 Raw MB | SQ8 Raw MB | FP32 Vec Storage | SQ8 Vec Storage | Vec Storage Red | FP32 Struct MB | SQ8 Struct MB | Total Struct Red | FP32 Heap MB | SQ8 Heap MB | Heap Delta Red |");
    System.out.println(
        "|-----------:|------------:|-----------:|-----------------:|----------------:|----------------:|---------------:|--------------:|-----------------:|-------------:|------------:|---------------:|");

    List<MemoryComparisonRow> memRows = new ArrayList<>();
    for (int n : scales) {
      MemoryComparisonRow row = evaluateMemory(n);
      memRows.add(row);
      System.out.println(row.toMarkdownRow());
    }

    System.out.println("\n### 2. Search Throughput and Latency Distribution (P50, P95, P99)");
    System.out.println(
        "|          N | Configuration            | Throughput (QPS) | Mean Latency (us) | P50 Latency (us) | P95 Latency (us) | P99 Latency (us) | Speedup vs FP32 |");
    System.out.println(
        "|-----------:|:-------------------------|-----------------:|------------------:|-----------------:|-----------------:|-----------------:|----------------:|");

    for (int n : scales) {
      FlatVsQuantizedFlatBenchmark benchmark = new FlatVsQuantizedFlatBenchmark();
      benchmark.initForTest(n);

      int numSamples = Math.min(2000, Math.max(200, 200000 / n));
      int numWarmup = Math.min(200, numSamples / 5);

      // Baseline measurements
      LatencyProfile fp32ScalarProf =
          benchmark.profile(Candidate.FP32_SCALAR, numWarmup, numSamples, 0);
      LatencyProfile fp32SimdProf =
          benchmark.profile(Candidate.FP32_SIMD, numWarmup, numSamples, 0);

      // Speedups relative to matching FP32 counterparts
      LatencyProfile sq8ScalarProf =
          benchmark.profile(
              Candidate.SQ8_SCALAR, numWarmup, numSamples, fp32ScalarProf.meanLatencyUs());
      LatencyProfile sq8SimdProf =
          benchmark.profile(
              Candidate.SQ8_SIMD, numWarmup, numSamples, fp32SimdProf.meanLatencyUs());

      System.out.println(fp32ScalarProf.toMarkdownRow());
      System.out.println(sq8ScalarProf.toMarkdownRow());
      System.out.println(fp32SimdProf.toMarkdownRow());
      System.out.println(sq8SimdProf.toMarkdownRow());
      System.out.println(
          "|------------|--------------------------|------------------|-------------------|------------------|------------------|------------------|-----------------|");
    }
  }

  public static void main(String[] args) {
    runSweep(DEFAULT_SCALES);
  }
}
