package com.nanovector.core.index;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.EpochVisitedSet;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.hnsw.LevelGenerator;
import com.nanovector.core.hnsw.NeighborSelector;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.quantization.QuantizedEuclideanDistance;
import com.nanovector.core.quantization.QuantizedVectorStorage;
import com.nanovector.core.storage.VectorDataView;
import com.nanovector.core.util.VectorUtils;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Approximate nearest-neighbor HNSW index backed by 8-bit scalar quantization (SQ8) and Asymmetric
 * Distance Computation (ADC).
 *
 * <p>Orchestrates {@link QuantizedVectorStorage} (compact 136 B/vec primitive byte storage at
 * $D=128$), {@link HnswGraph} (decoupled multi-layer graph topology), and {@link
 * QuantizedEuclideanDistance} (SIMD ADC distance kernel) to provide memory-compressed approximate
 * nearest-neighbor search.
 *
 * <p><b>Execution Characteristics:</b>
 *
 * <ul>
 *   <li><b>Storage Compression:</b> Replaces full-precision FP32 vector payloads ($D \times 4$
 *       bytes) with unsigned 8-bit integers plus per-vector affine metadata ($D + 8$ bytes).
 *   <li><b>SIMD ADC Distance Hot Path:</b> Evaluates distances from query vectors directly against
 *       contiguous byte buffers without intermediate vector array allocations.
 *   <li><b>Decoupled Topology:</b> Reuses {@link HnswGraph}, {@link HnswNode}, {@link
 *       NeighborSelector}, and {@link EpochVisitedSet} identically to {@link HnswIndex}.
 * </ul>
 */
public final class QuantizedHnswIndex implements VectorIndex {

  private final int dimension;
  private final DistanceMetric metric;
  private final QuantizedEuclideanDistance calculator;
  private final QuantizedVectorStorage storage;
  private final HnswGraph graph;
  private final HnswConfig config;
  private final LevelGenerator levelGenerator;
  private final EpochVisitedSet visitedSet;
  private final float[] nodeEvalBuffer;
  private final NeighborSelector.NodeDistanceEvaluator nodeEvaluator;

  public QuantizedHnswIndex(int dimension, DistanceMetric metric, HnswConfig config) {
    this(dimension, metric, config, 1024, true);
  }

  public QuantizedHnswIndex(
      int dimension, DistanceMetric metric, HnswConfig config, boolean useSimd) {
    this(dimension, metric, config, 1024, useSimd);
  }

  public QuantizedHnswIndex(
      int dimension, DistanceMetric metric, HnswConfig config, int initialCapacity) {
    this(dimension, metric, config, initialCapacity, true);
  }

  public QuantizedHnswIndex(
      int dimension,
      DistanceMetric metric,
      HnswConfig config,
      int initialCapacity,
      boolean useSimd) {
    this(dimension, metric, config, initialCapacity, QuantizedEuclideanDistance.create(useSimd));
  }

  public QuantizedHnswIndex(
      int dimension,
      DistanceMetric metric,
      HnswConfig config,
      int initialCapacity,
      QuantizedEuclideanDistance calculator) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (metric != DistanceMetric.EUCLIDEAN) {
      throw new UnsupportedOperationException(
          "QuantizedHnswIndex currently supports EUCLIDEAN metric only, got: " + metric);
    }
    this.dimension = dimension;
    this.metric = metric;
    this.config = Objects.requireNonNull(config, "HnswConfig must not be null");
    this.calculator = Objects.requireNonNull(calculator, "Calculator must not be null");
    this.storage = new QuantizedVectorStorage(dimension, initialCapacity);
    this.graph = new HnswGraph(config);
    this.levelGenerator = new LevelGenerator(config);
    this.visitedSet = new EpochVisitedSet(initialCapacity);
    this.nodeEvalBuffer = new float[dimension];
    this.nodeEvaluator =
        (a, b) -> {
          this.storage.copyVector(a, this.nodeEvalBuffer);
          int offsetB = b * dimension;
          return this.calculator.distance(
              this.storage.vectorBuffer(),
              offsetB,
              this.storage.getMin(b),
              this.storage.getScale(b),
              this.nodeEvalBuffer);
        };
  }

  /**
   * Constructs a QuantizedHnswIndex with pre-existing storage and graph.
   *
   * @param dimension vector dimension
   * @param metric distance metric (must be EUCLIDEAN)
   * @param config HNSW configuration
   * @param storage quantized vector storage
   * @param graph HNSW graph topology
   * @param calculator quantized distance calculator
   */
  public QuantizedHnswIndex(
      int dimension,
      DistanceMetric metric,
      HnswConfig config,
      QuantizedVectorStorage storage,
      HnswGraph graph,
      QuantizedEuclideanDistance calculator) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (metric != DistanceMetric.EUCLIDEAN) {
      throw new UnsupportedOperationException(
          "QuantizedHnswIndex currently supports EUCLIDEAN metric only, got: " + metric);
    }
    this.dimension = dimension;
    this.metric = metric;
    this.config = Objects.requireNonNull(config, "HnswConfig must not be null");
    this.calculator = Objects.requireNonNull(calculator, "Calculator must not be null");
    this.storage = Objects.requireNonNull(storage, "Storage must not be null");
    this.graph = Objects.requireNonNull(graph, "Graph must not be null");
    this.levelGenerator = new LevelGenerator(config);
    this.visitedSet = new EpochVisitedSet(Math.max(1024, storage.size() + 1));
    this.nodeEvalBuffer = new float[dimension];
    this.nodeEvaluator =
        (a, b) -> {
          this.storage.copyVector(a, this.nodeEvalBuffer);
          int offsetB = b * dimension;
          return this.calculator.distance(
              this.storage.vectorBuffer(),
              offsetB,
              this.storage.getMin(b),
              this.storage.getScale(b),
              this.nodeEvalBuffer);
        };
  }

  /**
   * Factory method to create a hybrid QuantizedHnswIndex from an existing FP32 {@link HnswIndex}.
   *
   * <p>Enables Strategy A (Hybrid): reuses the FP32-constructed reference graph topology while
   * quantizing vector storage into 8-bit compact representations for search acceleration.
   *
   * @param fp32Index the source FP32 HNSW index
   * @return new QuantizedHnswIndex backed by quantized storage and the FP32 graph topology
   */
  public static QuantizedHnswIndex fromFp32(HnswIndex fp32Index) {
    return fromFp32(fp32Index, true);
  }

  /**
   * Factory method to create a hybrid QuantizedHnswIndex from an existing FP32 {@link HnswIndex}
   * with configurable SIMD acceleration.
   */
  public static QuantizedHnswIndex fromFp32(HnswIndex fp32Index, boolean useSimd) {
    Objects.requireNonNull(fp32Index, "fp32Index must not be null");
    int dim = fp32Index.dimension();
    int size = fp32Index.size();
    QuantizedVectorStorage storage = new QuantizedVectorStorage(dim, Math.max(1024, size));
    float[] fp32Buffer = fp32Index.vectorData().vectorBuffer();
    long[] externalIds = fp32Index.vectorData().externalIdBuffer();
    float[] temp = new float[dim];
    for (int i = 0; i < size; i++) {
      System.arraycopy(fp32Buffer, i * dim, temp, 0, dim);
      storage.insert(externalIds[i], temp);
    }
    QuantizedEuclideanDistance calc = QuantizedEuclideanDistance.create(useSimd);
    return new QuantizedHnswIndex(
        dim, fp32Index.metric(), fp32Index.config(), storage, fp32Index.graph(), calc);
  }

  // ── VectorIndex contract ────────────────────────────────────────────

  @Override
  public synchronized void insert(long id, float[] vector) {
    VectorUtils.checkDimension(vector, dimension);
    VectorUtils.checkFinite(vector);

    // 1. Store quantized vector -> get internalId
    int internalId = storage.insert(id, vector);

    // 2. Generate random level
    int nodeLevel = levelGenerator.nextLevel();

    // 3. Create graph node
    HnswNode newNode = new HnswNode(internalId, nodeLevel);
    graph.addNode(newNode);

    // 4. Ensure visitedSet capacity
    visitedSet.ensureCapacity(internalId + 1);

    // 5. If this is the first node, nothing more to do
    if (graph.size() == 1) {
      return;
    }

    // Build distance evaluator from stored nodes to the newly inserted vector (FP32)
    HnswGraph.DistanceToQuery distanceToNew = buildDistanceToQuery(vector);
    NeighborSelector.NodeDistanceEvaluator nodeEval = nodeEvaluator;

    int currentEntryPoint = graph.entryPointId();
    int currentMaxLevel = graph.maxLevel();

    // 6. Phase 1: Greedy descent from current max level down to nodeLevel + 1
    for (int layer = currentMaxLevel; layer > nodeLevel; layer--) {
      if (graph.getNode(currentEntryPoint).maxLevel() >= layer) {
        currentEntryPoint = graph.greedyClosest(distanceToNew, currentEntryPoint, layer);
      }
    }

    // 7. Phase 2: searchLayer + connect at each layer from min(currentMaxLevel, nodeLevel) down to
    // 0
    int insertionTopLayer = Math.min(currentMaxLevel, nodeLevel);
    for (int layer = insertionTopLayer; layer >= 0; layer--) {
      visitedSet.nextEpoch();

      int ef = config.efConstruction();
      List<NeighborSelector.Candidate> candidates =
          graph.searchLayer(distanceToNew, new int[] {currentEntryPoint}, ef, layer, visitedSet);

      int maxDegree = graph.maxDegreeForLayer(layer);
      int[] selectedNeighbors =
          NeighborSelector.selectNeighbors(new ArrayList<>(candidates), maxDegree, nodeEval);

      for (int neighborId : selectedNeighbors) {
        graph.connect(internalId, neighborId, layer, nodeEval);
      }

      if (!candidates.isEmpty()) {
        currentEntryPoint = candidates.get(0).id();
      }
    }

    // 8. Update entry point if newly inserted node has higher level than current max
    if (nodeLevel > currentMaxLevel) {
      graph.setEntryPoint(internalId, nodeLevel);
    }
  }

  @Override
  public List<SearchResult> searchKnn(float[] query, int k) {
    return searchKnn(query, k, config.efSearch());
  }

  /**
   * Searches for the {@code k} approximate nearest neighbors with a custom {@code efSearch}
   * parameter using SIMD ADC distance evaluation.
   *
   * @param query query vector matching index dimension
   * @param k number of neighbors to return
   * @param efSearch size of the dynamic candidate list for this search
   * @return list of search results sorted ascending by distance
   */
  public List<SearchResult> searchKnn(float[] query, int k, int efSearch) {
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    if (efSearch <= 0) {
      throw new IllegalArgumentException("efSearch must be positive: " + efSearch);
    }
    VectorUtils.checkDimension(query, dimension);
    VectorUtils.checkFinite(query);

    if (storage.size() == 0) {
      return Collections.emptyList();
    }

    int actualK = Math.min(k, storage.size());
    int effectiveEf = Math.max(efSearch, actualK);

    HnswGraph.DistanceToQuery distanceToQuery = buildDistanceToQuery(query);

    int currentEntryPoint = graph.entryPointId();
    int currentMaxLevel = graph.maxLevel();

    // Phase 1: Greedy descent from max level down to layer 1
    for (int layer = currentMaxLevel; layer >= 1; layer--) {
      currentEntryPoint = graph.greedyClosest(distanceToQuery, currentEntryPoint, layer);
    }

    // Phase 2: searchLayer at layer 0 with effectiveEf
    visitedSet.nextEpoch();
    List<NeighborSelector.Candidate> candidates =
        graph.searchLayer(
            distanceToQuery, new int[] {currentEntryPoint}, effectiveEf, 0, visitedSet);

    int resultCount = Math.min(actualK, candidates.size());
    List<SearchResult> results = new ArrayList<>(resultCount);
    for (int i = 0; i < resultCount; i++) {
      NeighborSelector.Candidate c = candidates.get(i);
      long externalId = storage.getExternalId(c.id());
      results.add(new SearchResult(externalId, c.distance()));
    }
    return results;
  }

  @Override
  public int size() {
    return storage.size();
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public DistanceMetric metric() {
    return metric;
  }

  @Override
  public VectorDataView vectorData() {
    throw new UnsupportedOperationException(
        "QuantizedHnswIndex maintains contiguous quantized byte storage. Use storage() to access QuantizedVectorStorage.");
  }

  // ── Accessors ────────────────────────────────────────────────────────

  public QuantizedVectorStorage storage() {
    return storage;
  }

  public HnswGraph graph() {
    return graph;
  }

  public HnswConfig config() {
    return config;
  }

  public QuantizedEuclideanDistance calculator() {
    return calculator;
  }

  // ── Helper methods ───────────────────────────────────────────────────

  private HnswGraph.DistanceToQuery buildDistanceToQuery(float[] queryVector) {
    byte[] buffer = storage.vectorBuffer();
    return internalId -> {
      int offset = internalId * dimension;
      float min = storage.getMin(internalId);
      float scale = storage.getScale(internalId);
      return calculator.distance(buffer, offset, min, scale, queryVector);
    };
  }
}
