package com.nanovector.benchmark.memory;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.QuantizedHnswIndex;
import com.nanovector.core.storage.VectorDataView;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Characterizes measured heap usage delta and structural memory cost across index architectures
 * (Flat vs HNSW) and dataset scales ($N \in \{1\text{K}, 10\text{K}, 50\text{K}, 100\text{K}\}$).
 *
 * <p>Scientific distinction:
 *
 * <ul>
 *   <li><b>Measured Heap Usage Delta</b>: Empirical difference in JVM heap occupancy ({@code
 *       totalMemory() - freeMemory()}) observed before and after index construction, with multiple
 *       GC stabilization passes. Subject to JVM runtime factors (TLAB, padding, GC timing).
 *   <li><b>Structural Memory Estimate</b>: Analytical derivation based on NanoVector's concrete
 *       data structures (contiguous primitive buffers, object headers, reference pointers, and
 *       neighbor arrays) under 64-bit HotSpot JVM with Compressed OOPs.
 *   <li><b>Memory Amplification</b>: Ratio of total index memory cost over raw vector payload ($N
 *       \times D \times 4$ bytes), serving as the foundational baseline for future quantization
 *       studies.
 * </ul>
 */
public final class MemoryFootprintProfiler {

  // HotSpot 64-bit with Compressed OOPs (<32GB Heap) sizing constants
  private static final int OBJECT_HEADER_BYTES = 16; // 12B mark+class, padded to 8B boundary
  private static final int ARRAY_HEADER_BYTES = 16; // 8B mark + 4B class + 4B length
  private static final int REF_BYTES = 4; // 32-bit compressed reference
  private static final int HASHMAP_NODE_BYTES =
      32; // header (12) + hash (4) + key (4) + val (4) + next (4) -> 28 padded to 32
  private static final int BOXED_LONG_BYTES = 24; // header (12) + long (8) -> 20 padded to 24
  private static final int BOXED_INT_BYTES = 16; // header (12) + int (4) -> 16
  private static final int HASHMAP_ENTRY_TOTAL =
      HASHMAP_NODE_BYTES + BOXED_LONG_BYTES + BOXED_INT_BYTES; // ~72 bytes

  public record MemoryReport(
      String indexType,
      int vectorCount,
      int dimension,
      long rawVectorPayloadBytes,
      long externalIdPayloadBytes,
      long storageStructuralBytes,
      long graphStructuralBytes,
      long totalStructuralBytes,
      long measuredHeapDeltaBytes,
      double structuralBytesPerVector,
      double measuredBytesPerVector,
      double graphOverheadPerVector,
      double structuralAmplification,
      double measuredAmplification) {

    public String toFormattedString() {
      StringBuilder sb = new StringBuilder();
      sb.append(
          String.format(
              "=== %s Memory Characterization (N=%,d, D=%d) ===%n",
              indexType, vectorCount, dimension));
      sb.append(
          String.format(
              "  Raw Vector Payload:           %8.2f MiB (%d bytes)%n",
              rawVectorPayloadBytes / (1024.0 * 1024.0), rawVectorPayloadBytes));
      sb.append(
          String.format(
              "  External IDs Payload:         %8.2f MiB (%d bytes)%n",
              externalIdPayloadBytes / (1024.0 * 1024.0), externalIdPayloadBytes));
      sb.append(
          String.format(
              "  Storage Structural Estimate:  %8.2f MiB (%d bytes)%n",
              storageStructuralBytes / (1024.0 * 1024.0), storageStructuralBytes));
      if (graphStructuralBytes > 0) {
        sb.append(
            String.format(
                "  Graph Structural Estimate:    %8.2f MiB (%d bytes)%n",
                graphStructuralBytes / (1024.0 * 1024.0), graphStructuralBytes));
        sb.append(
            String.format(
                "  Graph Overhead / Vector:      %8.1f bytes/vector%n", graphOverheadPerVector));
      }
      sb.append(
          String.format(
              "  Total Structural Estimate:    %8.2f MiB (%d bytes)%n",
              totalStructuralBytes / (1024.0 * 1024.0), totalStructuralBytes));
      sb.append(
          String.format(
              "  Measured Heap Usage Delta:    %8.2f MiB (%d bytes)%n",
              measuredHeapDeltaBytes / (1024.0 * 1024.0), measuredHeapDeltaBytes));
      sb.append(
          String.format(
              "  Structural Bytes / Vector:    %8.1f bytes/vector%n", structuralBytesPerVector));
      sb.append(
          String.format(
              "  Measured Bytes / Vector:      %8.1f bytes/vector%n", measuredBytesPerVector));
      sb.append(
          String.format(
              "  Structural Amplification:     %8.2fx (relative to raw vector payload)%n",
              structuralAmplification));
      sb.append(
          String.format(
              "  Measured Amplification:       %8.2fx (relative to raw vector payload)%n",
              measuredAmplification));
      return sb.toString();
    }
  }

  private MemoryFootprintProfiler() {}

  /** Profiles memory footprint for FlatIndex at the given scale. */
  public static MemoryReport profileFlat(int vectorCount, int dimension) {
    long rawPayload = (long) vectorCount * dimension * Float.BYTES;
    long externalIdPayload = (long) vectorCount * Long.BYTES;

    Random rng = new Random(42L);
    float[][] dataset = generateDataset(vectorCount, dimension, rng);

    // Measure empirical heap delta
    FlatIndex[] holder = new FlatIndex[1];
    long heapDelta =
        measureHeapDelta(
            () -> {
              FlatIndex index =
                  new FlatIndex(dimension, DistanceMetric.EUCLIDEAN, vectorCount, true);
              for (int i = 0; i < vectorCount; i++) {
                index.insert(i, dataset[i]);
              }
              holder[0] = index;
            });

    FlatIndex index = holder[0];
    long storageStructural = estimateStorageStructural(index.vectorData(), vectorCount, dimension);
    long totalStructural = storageStructural + OBJECT_HEADER_BYTES + (2 * REF_BYTES);

    return createReport(
        "FlatIndex",
        vectorCount,
        dimension,
        rawPayload,
        externalIdPayload,
        storageStructural,
        0L,
        totalStructural,
        heapDelta);
  }

  /** Profiles memory footprint for QuantizedFlatIndex (SQ8) at the given scale. */
  public static MemoryReport profileQuantizedFlat(int vectorCount, int dimension) {
    long rawPayload = (long) vectorCount * dimension * Float.BYTES;
    long externalIdPayload = (long) vectorCount * Long.BYTES;

    Random rng = new Random(42L);
    float[][] dataset = generateDataset(vectorCount, dimension, rng);

    // Measure empirical heap delta
    com.nanovector.core.index.QuantizedFlatIndex[] holder =
        new com.nanovector.core.index.QuantizedFlatIndex[1];
    long heapDelta =
        measureHeapDelta(
            () -> {
              com.nanovector.core.index.QuantizedFlatIndex index =
                  new com.nanovector.core.index.QuantizedFlatIndex(dimension, vectorCount, true);
              for (int i = 0; i < vectorCount; i++) {
                index.insert(i, dataset[i]);
              }
              holder[0] = index;
            });

    com.nanovector.core.index.QuantizedFlatIndex index = holder[0];
    long storageStructural =
        estimateQuantizedStorageStructural(index.storage(), vectorCount, dimension);
    long totalStructural = storageStructural + OBJECT_HEADER_BYTES + (2 * REF_BYTES);

    return createReport(
        "QuantizedFlatIndex",
        vectorCount,
        dimension,
        rawPayload,
        externalIdPayload,
        storageStructural,
        0L,
        totalStructural,
        heapDelta);
  }

  /** Profiles memory footprint for HnswIndex at the given scale. */
  public static MemoryReport profileHnsw(int vectorCount, int dimension, HnswConfig config) {
    long rawPayload = (long) vectorCount * dimension * Float.BYTES;
    long externalIdPayload = (long) vectorCount * Long.BYTES;

    Random rng = new Random(42L);
    float[][] dataset = generateDataset(vectorCount, dimension, rng);

    // Measure empirical heap delta
    HnswIndex[] holder = new HnswIndex[1];
    long heapDelta =
        measureHeapDelta(
            () -> {
              HnswIndex index =
                  new HnswIndex(dimension, DistanceMetric.EUCLIDEAN, config, vectorCount, true);
              for (int i = 0; i < vectorCount; i++) {
                index.insert(i, dataset[i]);
              }
              holder[0] = index;
            });

    HnswIndex index = holder[0];
    long storageStructural = estimateStorageStructural(index.vectorData(), vectorCount, dimension);
    long graphStructural = estimateGraphStructural(index.graph(), vectorCount);
    long visitedSetStructural =
        ARRAY_HEADER_BYTES + align8((long) vectorCount * Integer.BYTES) + OBJECT_HEADER_BYTES;
    long totalGraph = graphStructural + visitedSetStructural;
    long totalStructural = storageStructural + totalGraph + OBJECT_HEADER_BYTES + (4 * REF_BYTES);

    return createReport(
        "HnswIndex",
        vectorCount,
        dimension,
        rawPayload,
        externalIdPayload,
        storageStructural,
        totalGraph,
        totalStructural,
        heapDelta);
  }

  /** Profiles memory footprint for QuantizedHnswIndex at the given scale. */
  public static MemoryReport profileQuantizedHnsw(
      int vectorCount, int dimension, HnswConfig config) {
    long rawPayload = (long) vectorCount * dimension * Float.BYTES;
    long externalIdPayload = (long) vectorCount * Long.BYTES;

    Random rng = new Random(42L);
    float[][] dataset = generateDataset(vectorCount, dimension, rng);

    // Measure empirical heap delta
    QuantizedHnswIndex[] holder = new QuantizedHnswIndex[1];
    long heapDelta =
        measureHeapDelta(
            () -> {
              QuantizedHnswIndex index =
                  new QuantizedHnswIndex(
                      dimension, DistanceMetric.EUCLIDEAN, config, vectorCount, true);
              for (int i = 0; i < vectorCount; i++) {
                index.insert(i, dataset[i]);
              }
              holder[0] = index;
            });

    QuantizedHnswIndex index = holder[0];
    long storageStructural =
        estimateQuantizedStorageStructural(index.storage(), vectorCount, dimension);
    long graphStructural = estimateGraphStructural(index.graph(), vectorCount);
    long visitedSetStructural =
        ARRAY_HEADER_BYTES + align8((long) vectorCount * Integer.BYTES) + OBJECT_HEADER_BYTES;
    long totalGraph = graphStructural + visitedSetStructural;
    long totalStructural = storageStructural + totalGraph + OBJECT_HEADER_BYTES + (4 * REF_BYTES);

    return createReport(
        "QuantizedHnswIndex",
        vectorCount,
        dimension,
        rawPayload,
        externalIdPayload,
        storageStructural,
        totalGraph,
        totalStructural,
        heapDelta);
  }

  /** Computes the structural memory estimate of VectorStorage via VectorDataView. */
  public static long estimateStorageStructural(VectorDataView view, int count, int dimension) {
    int capacity = Math.max(count, view.vectorBuffer().length / dimension);

    // 1. Primitive float buffer: float[capacity * dimension]
    long vectorArrayBytes = ARRAY_HEADER_BYTES + align8((long) capacity * dimension * Float.BYTES);

    // 2. Primitive long external IDs: long[capacity]
    long externalIdArrayBytes = ARRAY_HEADER_BYTES + align8((long) capacity * Long.BYTES);

    // 3. HashMap<Long, Integer> externalToInternal:
    // Capacity of internal bucket table (next power of 2 >= count / 0.75)
    int tableCapacity = Integer.highestOneBit(Math.max(16, (int) (count / 0.75f))) << 1;
    long tableArrayBytes = ARRAY_HEADER_BYTES + align8((long) tableCapacity * REF_BYTES);
    long mapEntriesBytes = (long) count * HASHMAP_ENTRY_TOTAL;
    long mapObjectBytes = OBJECT_HEADER_BYTES + (4 * REF_BYTES) + 8; // fields

    long totalMap = mapObjectBytes + tableArrayBytes + mapEntriesBytes;

    // VectorStorage object itself
    long storageObjBytes = OBJECT_HEADER_BYTES + (3 * REF_BYTES) + (2 * Integer.BYTES);

    return align8(storageObjBytes + vectorArrayBytes + externalIdArrayBytes + totalMap);
  }

  /** Computes the structural memory estimate of QuantizedVectorStorage. */
  public static long estimateQuantizedStorageStructural(
      com.nanovector.core.quantization.QuantizedVectorStorage storage, int count, int dimension) {
    int capacity = Math.max(count, storage.vectorBuffer().length / dimension);

    // 1. Primitive byte buffer: byte[capacity * dimension]
    long vectorArrayBytes = ARRAY_HEADER_BYTES + align8((long) capacity * dimension * Byte.BYTES);

    // 2. Parallel primitive float buffers for per-vector parameters: mins and scales
    long minsArrayBytes = ARRAY_HEADER_BYTES + align8((long) capacity * Float.BYTES);
    long scalesArrayBytes = ARRAY_HEADER_BYTES + align8((long) capacity * Float.BYTES);

    // 3. Primitive long external IDs: long[capacity]
    long externalIdArrayBytes = ARRAY_HEADER_BYTES + align8((long) capacity * Long.BYTES);

    // 4. HashMap<Long, Integer> externalToInternal:
    int tableCapacity = Integer.highestOneBit(Math.max(16, (int) (count / 0.75f))) << 1;
    long tableArrayBytes = ARRAY_HEADER_BYTES + align8((long) tableCapacity * REF_BYTES);
    long mapEntriesBytes = (long) count * HASHMAP_ENTRY_TOTAL;
    long mapObjectBytes = OBJECT_HEADER_BYTES + (4 * REF_BYTES) + 8; // fields

    long totalMap = mapObjectBytes + tableArrayBytes + mapEntriesBytes;

    // QuantizedVectorStorage object itself: header + 5 refs (quantizer, vectors, mins, scales,
    // externalIds, map) + 2 ints (dimension, size)
    long storageObjBytes = OBJECT_HEADER_BYTES + (6 * REF_BYTES) + (2 * Integer.BYTES);

    return align8(
        storageObjBytes
            + vectorArrayBytes
            + minsArrayBytes
            + scalesArrayBytes
            + externalIdArrayBytes
            + totalMap);
  }

  /** Computes the structural memory estimate of HnswGraph (nodes and neighbor adjacency lists). */
  public static long estimateGraphStructural(HnswGraph graph, int count) {
    // 1. HnswGraph object: header + fields
    long graphObjBytes = OBJECT_HEADER_BYTES + (3 * Integer.BYTES) + (2 * REF_BYTES);

    // 2. ArrayList<HnswNode> elementData: Object[capacity]
    long arrayListBytes =
        OBJECT_HEADER_BYTES
            + (2 * Integer.BYTES)
            + REF_BYTES
            + ARRAY_HEADER_BYTES
            + align8((long) count * REF_BYTES);

    // 3. Nodes and multi-layer adjacency lists
    long nodesTotalBytes = 0;
    for (int i = 0; i < count; i++) {
      HnswNode node = graph.getNode(i);
      int maxLevel = node.maxLevel();

      // HnswNode object: header + internalId (4) + maxLevel (4) + neighbors ref (4) -> 24 bytes
      long nodeObjBytes = OBJECT_HEADER_BYTES + (2 * Integer.BYTES) + REF_BYTES;

      // int[maxLevel + 1][] array of layer references
      long layerRefArrayBytes = ARRAY_HEADER_BYTES + align8((long) (maxLevel + 1) * REF_BYTES);

      // int[] arrays for each layer
      long layerArraysBytes = 0;
      for (int l = 0; l <= maxLevel; l++) {
        int degree = node.degree(l);
        layerArraysBytes += ARRAY_HEADER_BYTES + align8((long) degree * Integer.BYTES);
      }

      nodesTotalBytes += align8(nodeObjBytes + layerRefArrayBytes + layerArraysBytes);
    }

    return graphObjBytes + arrayListBytes + nodesTotalBytes;
  }

  /**
   * Measures the empirical heap occupancy delta (totalMemory - freeMemory) observed before and
   * after task execution with multiple GC stabilization passes.
   */
  public static long measureHeapDelta(Runnable task) {
    stabilizeGc();
    long before = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

    task.run();

    stabilizeGc();
    long after = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

    return Math.max(0, after - before);
  }

  private static void stabilizeGc() {
    for (int i = 0; i < 4; i++) {
      System.gc();
      try {
        Thread.sleep(50);
      } catch (InterruptedException ignored) {
        Thread.currentThread().interrupt();
      }
    }
  }

  private static long align8(long bytes) {
    return (bytes + 7) & ~7;
  }

  private static float[][] generateDataset(int count, int dim, Random rng) {
    float[][] data = new float[count][dim];
    for (int i = 0; i < count; i++) {
      for (int d = 0; d < dim; d++) {
        data[i][d] = rng.nextFloat() * 2.0f - 1.0f;
      }
    }
    return data;
  }

  private static MemoryReport createReport(
      String indexType,
      int count,
      int dim,
      long rawPayload,
      long externalIdPayload,
      long storageStructural,
      long graphStructural,
      long totalStructural,
      long heapDelta) {
    double structuralPerVec = (double) totalStructural / count;
    double measuredPerVec = (double) heapDelta / count;
    double graphPerVec = (double) graphStructural / count;
    double structAmp = (double) totalStructural / rawPayload;
    double measAmp = (double) heapDelta / rawPayload;

    return new MemoryReport(
        indexType,
        count,
        dim,
        rawPayload,
        externalIdPayload,
        storageStructural,
        graphStructural,
        totalStructural,
        heapDelta,
        structuralPerVec,
        measuredPerVec,
        graphPerVec,
        structAmp,
        measAmp);
  }

  /** Record capturing comparative memory breakdown between FP32 HNSW and Pure SQ8 HNSW. */
  public record HnswMemoryComparisonRow(
      int n,
      int dimension,
      double fp32RawVectorMiB,
      double fp32VectorStorageMiB,
      double fp32GraphTopologyMiB,
      double fp32TotalStructuralMiB,
      double fp32MeasuredHeapMiB,
      double fp32StructuralBytesPerVec,
      double fp32MeasuredBytesPerVec,
      double sq8VectorStorageMiB,
      double sq8GraphTopologyMiB,
      double sq8TotalStructuralMiB,
      double sq8MeasuredHeapMiB,
      double sq8StructuralBytesPerVec,
      double sq8MeasuredBytesPerVec,
      double vectorStorageReduction,
      double totalStructuralReduction,
      double measuredHeapReduction,
      double sq8GraphTopologyFraction) {

    public String toBreakdownMarkdownRow() {
      return String.format(
          "| %,10d | %8.2f MiB | %10.2f MiB | %10.2f MiB | %10.2f MiB | %9.2f MiB | %10.2f MiB | %10.2f MiB | %10.2f MiB | %9.2f MiB |",
          n,
          fp32RawVectorMiB,
          fp32VectorStorageMiB,
          fp32GraphTopologyMiB,
          fp32TotalStructuralMiB,
          fp32MeasuredHeapMiB,
          sq8VectorStorageMiB,
          sq8GraphTopologyMiB,
          sq8TotalStructuralMiB,
          sq8MeasuredHeapMiB);
    }

    public String toComparisonMarkdownRow() {
      return String.format(
          "| %,10d | %10.1f B/v | %10.1f B/v | %9.1f B/v | %9.1f B/v | %11.2fx | %11.2fx | %10.2fx | %12.1f%% |",
          n,
          fp32StructuralBytesPerVec,
          fp32MeasuredBytesPerVec,
          sq8StructuralBytesPerVec,
          sq8MeasuredBytesPerVec,
          vectorStorageReduction,
          totalStructuralReduction,
          measuredHeapReduction,
          sq8GraphTopologyFraction);
    }
  }

  /** Compares memory footprint between FP32 HNSW and Pure SQ8 HNSW at a specific scale. */
  public static HnswMemoryComparisonRow compareHnswMemory(int count, int dim, HnswConfig config) {
    MemoryReport fp32 = profileHnsw(count, dim, config);
    MemoryReport sq8 = profileQuantizedHnsw(count, dim, config);

    double toMiB = 1024.0 * 1024.0;
    double fp32Raw = fp32.rawVectorPayloadBytes() / toMiB;
    double fp32Storage = fp32.storageStructuralBytes() / toMiB;
    double fp32Graph = fp32.graphStructuralBytes() / toMiB;
    double fp32Total = fp32.totalStructuralBytes() / toMiB;
    double fp32Heap = fp32.measuredHeapDeltaBytes() / toMiB;

    double sq8Storage = sq8.storageStructuralBytes() / toMiB;
    double sq8Graph = sq8.graphStructuralBytes() / toMiB;
    double sq8Total = sq8.totalStructuralBytes() / toMiB;
    double sq8Heap = sq8.measuredHeapDeltaBytes() / toMiB;

    double storageRed =
        (double) fp32.storageStructuralBytes() / Math.max(1L, sq8.storageStructuralBytes());
    double totalRed =
        (double) fp32.totalStructuralBytes() / Math.max(1L, sq8.totalStructuralBytes());
    double heapRed =
        (double) fp32.measuredHeapDeltaBytes() / Math.max(1L, sq8.measuredHeapDeltaBytes());
    double graphFrac =
        (double) sq8.graphStructuralBytes() / Math.max(1.0, sq8.totalStructuralBytes()) * 100.0;

    return new HnswMemoryComparisonRow(
        count,
        dim,
        fp32Raw,
        fp32Storage,
        fp32Graph,
        fp32Total,
        fp32Heap,
        fp32.structuralBytesPerVector(),
        fp32.measuredBytesPerVector(),
        sq8Storage,
        sq8Graph,
        sq8Total,
        sq8Heap,
        sq8.structuralBytesPerVector(),
        sq8.measuredBytesPerVector(),
        storageRed,
        totalRed,
        heapRed,
        graphFrac);
  }

  /** Runs a comparative memory sweep across multiple scales. */
  public static List<HnswMemoryComparisonRow> runHnswMemorySweep(
      int[] scales, int dimension, HnswConfig config) {
    List<HnswMemoryComparisonRow> rows = new ArrayList<>();
    for (int n : scales) {
      System.out.printf("Profiling memory footprint for Scale N = %,d (D=%d)...\n", n, dimension);
      System.out.flush();
      HnswMemoryComparisonRow row = compareHnswMemory(n, dimension, config);
      rows.add(row);
      System.out.printf(
          "  Scale N = %,d | FP32 Struct: %.2f MiB (Heap: %.2f MiB) | SQ8 Struct: %.2f MiB (Heap: %.2f MiB) | Reduction: %.2fx (Struct), %.2fx (Heap) | Graph Frac: %.1f%%\n",
          n,
          row.fp32TotalStructuralMiB(),
          row.fp32MeasuredHeapMiB(),
          row.sq8TotalStructuralMiB(),
          row.sq8MeasuredHeapMiB(),
          row.totalStructuralReduction(),
          row.measuredHeapReduction(),
          row.sq8GraphTopologyFraction());
      System.out.flush();
    }
    return rows;
  }

  public static void printMemoryReportTables(List<HnswMemoryComparisonRow> rows) {
    System.out.println(
        "====================================================================================================================");
    System.out.println(
        "                 PHASE 6C: HNSW MEMORY FOOTPRINT CHARACTERIZATION (FP32 HNSW vs PURE SQ8 HNSW)                       ");
    System.out.println(
        "====================================================================================================================");

    System.out.println(
        "### Table 1: Structural and Measured Heap Memory Breakdown across Scales (D=128)");
    System.out.println(
        "|          N |   Raw Vec | FP32 Storage |  FP32 Graph |  FP32 Total |  FP32 Heap |  SQ8 Storage |   SQ8 Graph |   SQ8 Total |   SQ8 Heap |");
    System.out.println(
        "|-----------:|----------:|-------------:|------------:|------------:|-----------:|-------------:|------------:|------------:|-----------:|");
    for (HnswMemoryComparisonRow r : rows) {
      System.out.println(r.toBreakdownMarkdownRow());
    }
    System.out.println();

    System.out.println("### Table 2: Memory Reduction Ratios and Graph Topology Dominance (D=128)");
    System.out.println(
        "|          N | FP32 Struct |  FP32 Heap |  SQ8 Struct |   SQ8 Heap | Storage Red | Total Struct Red |  Heap Red | Topology Frac |");
    System.out.println(
        "|-----------:|------------:|-----------:|------------:|-----------:|------------:|-----------------:|----------:|--------------:|");
    for (HnswMemoryComparisonRow r : rows) {
      System.out.println(r.toComparisonMarkdownRow());
    }
    System.out.println(
        "====================================================================================================================\n");
  }

  public static void main(String[] args) {
    int[] scales = {1000, 10000, 50000, 100000};
    int dim = 128;
    HnswConfig config = HnswConfig.withSeed(42L).withEfSearch(50);

    if (args.length > 0 && "--flat-only".equals(args[0])) {
      System.out.println(
          "==========================================================================");
      System.out.println("   NANOVECTOR MEMORY CHARACTERIZATION: FLAT vs HNSW (D=128)");
      System.out.println(
          "==========================================================================\n");

      for (int n : scales) {
        MemoryReport flatRep = profileFlat(n, dim);
        System.out.println(flatRep.toFormattedString());

        MemoryReport hnswRep = profileHnsw(n, dim, config);
        System.out.println(hnswRep.toFormattedString());
        System.out.println(
            "--------------------------------------------------------------------------\n");
      }
      return;
    }

    List<HnswMemoryComparisonRow> rows = runHnswMemorySweep(scales, dim, config);
    printMemoryReportTables(rows);
  }
}
