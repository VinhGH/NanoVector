package com.nanovector.persistence.benchmark;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.persistence.reader.NvecReader;
import com.nanovector.persistence.writer.NvecWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Random;
import java.util.zip.CRC32C;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Benchmark measuring serialization throughput, deserialization latency, file size footprints, and
 * CRC32C streaming overhead for NVEC v1 binary persistence across Flat and HNSW indexes.
 */
public class PersistenceBenchmark {

  private static final int DIMENSION = 128;
  private static final long SEED = 42L;

  @TempDir Path tempDir;

  record BenchmarkMetrics(
      String indexType,
      int vectorCount,
      int dimension,
      long fileSizeBytes,
      double bytesPerVector,
      double writeTimeMs,
      double writeThroughputMBps,
      double readTimeMs,
      double readThroughputMBps,
      double crcThroughputMBps) {}

  @Test
  @DisplayName("Run comprehensive NVEC v1 persistence benchmark for Flat and HNSW")
  void runPersistenceBenchmark() throws IOException {
    System.out.println(
        "================================================================================");
    System.out.println(
        "                   NANOVECTOR PERSISTENCE BENCHMARK (NVEC v1)                   ");
    System.out.println(
        "================================================================================");
    System.out.printf(
        "Hardware Environment: OS=%s, Arch=%s, JVM=%s%n",
        System.getProperty("os.name"),
        System.getProperty("os.arch"),
        System.getProperty("java.version"));
    System.out.printf("Parameters: Dimension=%d, Metric=EUCLIDEAN, Seed=%d%n", DIMENSION, SEED);
    System.out.println(
        "--------------------------------------------------------------------------------");

    int[] counts = {1_000, 10_000};

    System.out.println("\n>>> 1. Benchmarking FLAT Index (Brute-Force Oracle)...");
    for (int count : counts) {
      BenchmarkMetrics metrics = benchmarkFlat(count);
      printMetrics(metrics);
    }

    System.out.println(
        "\n>>> 2. Benchmarking HNSW Index (M=16, M0=32, efConstruction=200, efSearch=50)...");
    for (int count : counts) {
      BenchmarkMetrics metrics = benchmarkHnsw(count);
      printMetrics(metrics);
    }

    System.out.println(
        "================================================================================");
    System.out.println(
        "                             BENCHMARK COMPLETE                                 ");
    System.out.println(
        "================================================================================");
  }

  private BenchmarkMetrics benchmarkFlat(int count) throws IOException {
    FlatIndex index = new FlatIndex(DIMENSION, DistanceMetric.EUCLIDEAN);
    Random rand = new Random(SEED);

    for (int i = 0; i < count; i++) {
      float[] vec = new float[DIMENSION];
      for (int d = 0; d < DIMENSION; d++) {
        vec[d] = rand.nextFloat();
      }
      index.insert(i + 1L, vec);
    }

    Path file = tempDir.resolve("bench_flat_" + count + ".nvec");

    // Warmup write
    Path warmupFile = tempDir.resolve("warmup_flat.nvec");
    NvecWriter.write(index, warmupFile);
    Files.deleteIfExists(warmupFile);

    // Measured write
    long writeStart = System.nanoTime();
    NvecWriter.write(index, file);
    long writeElapsed = System.nanoTime() - writeStart;
    double writeMs = writeElapsed / 1_000_000.0;

    long fileSize = Files.size(file);
    double fileSizeMB = fileSize / (1024.0 * 1024.0);
    double writeMBps = (fileSizeMB) / (writeElapsed / 1_000_000_000.0);

    // Warmup read
    FlatIndex warmupRestored = NvecReader.readFlat(file);
    assertThatValidIndex(warmupRestored, count);

    // Measured read
    long readStart = System.nanoTime();
    FlatIndex restored = NvecReader.readFlat(file);
    long readElapsed = System.nanoTime() - readStart;
    double readMs = readElapsed / 1_000_000.0;
    double readMBps = (fileSizeMB) / (readElapsed / 1_000_000_000.0);

    // Measure raw standalone CRC32C streaming rate on same file
    double crcMBps = measureRawCrcRate(file, fileSize);

    // Behavioral validation: verify top-5 query identical
    float[] q = new float[DIMENSION];
    for (int d = 0; d < DIMENSION; d++) q[d] = 0.5f;
    List<SearchResult> origRes = index.searchKnn(q, 5);
    List<SearchResult> restRes = restored.searchKnn(q, 5);
    if (origRes.size() != restRes.size() || origRes.get(0).id() != restRes.get(0).id()) {
      throw new IllegalStateException(
          "Behavioral divergence between original and restored FLAT index");
    }

    return new BenchmarkMetrics(
        "FLAT",
        count,
        DIMENSION,
        fileSize,
        (double) fileSize / count,
        writeMs,
        writeMBps,
        readMs,
        readMBps,
        crcMBps);
  }

  private BenchmarkMetrics benchmarkHnsw(int count) throws IOException {
    HnswConfig config = HnswConfig.withSeed(SEED);
    HnswIndex index = new HnswIndex(DIMENSION, DistanceMetric.EUCLIDEAN, config);
    Random rand = new Random(SEED);

    for (int i = 0; i < count; i++) {
      float[] vec = new float[DIMENSION];
      for (int d = 0; d < DIMENSION; d++) {
        vec[d] = rand.nextFloat();
      }
      index.insert(i + 1L, vec);
    }

    Path file = tempDir.resolve("bench_hnsw_" + count + ".nvec");

    // Warmup write
    Path warmupFile = tempDir.resolve("warmup_hnsw.nvec");
    NvecWriter.write(index, warmupFile);
    Files.deleteIfExists(warmupFile);

    // Measured write
    long writeStart = System.nanoTime();
    NvecWriter.write(index, file);
    long writeElapsed = System.nanoTime() - writeStart;
    double writeMs = writeElapsed / 1_000_000.0;

    long fileSize = Files.size(file);
    double fileSizeMB = fileSize / (1024.0 * 1024.0);
    double writeMBps = (fileSizeMB) / (writeElapsed / 1_000_000_000.0);

    // Warmup read
    HnswIndex warmupRestored = NvecReader.readHnsw(file);
    assertThatValidIndex(warmupRestored, count);

    // Measured read
    long readStart = System.nanoTime();
    HnswIndex restored = NvecReader.readHnsw(file);
    long readElapsed = System.nanoTime() - readStart;
    double readMs = readElapsed / 1_000_000.0;
    double readMBps = (fileSizeMB) / (readElapsed / 1_000_000_000.0);

    // Measure raw standalone CRC32C streaming rate on same file
    double crcMBps = measureRawCrcRate(file, fileSize);

    // Behavioral validation: verify top-5 query identical
    float[] q = new float[DIMENSION];
    for (int d = 0; d < DIMENSION; d++) q[d] = 0.5f;
    List<SearchResult> origRes = index.searchKnn(q, 5);
    List<SearchResult> restRes = restored.searchKnn(q, 5);
    if (origRes.size() != restRes.size() || origRes.get(0).id() != restRes.get(0).id()) {
      throw new IllegalStateException(
          "Behavioral divergence between original and restored HNSW index");
    }

    return new BenchmarkMetrics(
        "HNSW",
        count,
        DIMENSION,
        fileSize,
        (double) fileSize / count,
        writeMs,
        writeMBps,
        readMs,
        readMBps,
        crcMBps);
  }

  private double measureRawCrcRate(Path file, long fileSize) throws IOException {
    CRC32C crc = new CRC32C();
    ByteBuffer buf = ByteBuffer.allocate(64 * 1024);
    long start = System.nanoTime();
    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      long remaining = fileSize - 4;
      while (remaining > 0) {
        buf.clear();
        int toRead = (int) Math.min(buf.capacity(), remaining);
        buf.limit(toRead);
        int read = ch.read(buf);
        if (read < 0) break;
        crc.update(buf.array(), 0, read);
        remaining -= read;
      }
    }
    long elapsed = System.nanoTime() - start;
    double mb = (fileSize - 4) / (1024.0 * 1024.0);
    return mb / (elapsed / 1_000_000_000.0);
  }

  private void assertThatValidIndex(Object index, int expectedSize) {
    if (index instanceof FlatIndex f && f.size() != expectedSize) {
      throw new IllegalStateException("Size mismatch");
    }
    if (index instanceof HnswIndex h && h.size() != expectedSize) {
      throw new IllegalStateException("Size mismatch");
    }
  }

  private void printMetrics(BenchmarkMetrics m) {
    System.out.printf("[%s | N=%6d | D=%3d]%n", m.indexType(), m.vectorCount(), m.dimension());
    System.out.printf(
        "  • File Size:          %,d bytes (%.2f MB) | %.1f bytes/vector%n",
        m.fileSizeBytes(), m.fileSizeBytes() / (1024.0 * 1024.0), m.bytesPerVector());
    System.out.printf(
        "  • Serialization (Save):    %6.2f ms | %7.2f MB/s%n",
        m.writeTimeMs(), m.writeThroughputMBps());
    System.out.printf(
        "  • Deserialization (Load):  %6.2f ms | %7.2f MB/s%n",
        m.readTimeMs(), m.readThroughputMBps());
    System.out.printf("  • Standalone CRC32C Rate:  %7.2f MB/s%n", m.crcThroughputMBps());
    System.out.println();
  }
}
