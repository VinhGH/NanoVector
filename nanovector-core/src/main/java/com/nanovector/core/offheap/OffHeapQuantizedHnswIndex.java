package com.nanovector.core.offheap;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.EpochVisitedSet;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.LevelGenerator;
import com.nanovector.core.hnsw.NeighborSelector;
import com.nanovector.core.index.VectorIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.storage.VectorStorage;
import com.nanovector.core.util.VectorUtils;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * High-performance off-heap HNSW index backed by 8-bit scalar quantization (SQ8) and direct native
 * SIMD Asymmetric Distance Computation (ADC).
 *
 * <p>Orchestrates {@link OffHeapQuantizedVectorStorage} (contiguous native 136 B/vec SQ8 payload at
 * $D=128$), {@link OffHeapHnswGraph} (zero-heap-object multi-layer graph topology), and {@link
 * OffHeapQuantizedEuclideanDistance} (native SIMD Vector API ADC kernel).
 *
 * <p>Implements both single-phase quantized search and two-phase candidate exploration with exact
 * FP32 re-ranking.
 */
public final class OffHeapQuantizedHnswIndex implements VectorIndex, AutoCloseable {

  private final int dimension;
  private final DistanceMetric metric;
  private final OffHeapQuantizedEuclideanDistance calculator;
  private final OffHeapQuantizedVectorStorage storage;
  private final OffHeapHnswGraph graph;
  private final HnswConfig config;
  private final LevelGenerator levelGenerator;
  private final EpochVisitedSet visitedSet;
  private final float[] nodeEvalBuffer;
  private final NeighborSelector.NodeDistanceEvaluator nodeEvaluator;

  public OffHeapQuantizedHnswIndex(int dimension, DistanceMetric metric, HnswConfig config) {
    this(dimension, metric, config, 1024, true);
  }

  public OffHeapQuantizedHnswIndex(
      int dimension, DistanceMetric metric, HnswConfig config, boolean useSimd) {
    this(dimension, metric, config, 1024, useSimd);
  }

  public OffHeapQuantizedHnswIndex(
      int dimension, DistanceMetric metric, HnswConfig config, int initialCapacity) {
    this(dimension, metric, config, initialCapacity, true);
  }

  public OffHeapQuantizedHnswIndex(
      int dimension,
      DistanceMetric metric,
      HnswConfig config,
      int initialCapacity,
      boolean useSimd) {
    this(
        dimension,
        metric,
        config,
        initialCapacity,
        OffHeapQuantizedEuclideanDistance.create(useSimd));
  }

  public OffHeapQuantizedHnswIndex(
      int dimension,
      DistanceMetric metric,
      HnswConfig config,
      int initialCapacity,
      OffHeapQuantizedEuclideanDistance calculator) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (metric != DistanceMetric.EUCLIDEAN) {
      throw new UnsupportedOperationException(
          "OffHeapQuantizedHnswIndex currently supports EUCLIDEAN metric only, got: " + metric);
    }
    this.dimension = dimension;
    this.metric = metric;
    this.config = Objects.requireNonNull(config, "HnswConfig must not be null");
    this.calculator = Objects.requireNonNull(calculator, "Calculator must not be null");
    this.storage = new OffHeapQuantizedVectorStorage(dimension, initialCapacity);
    this.graph = new OffHeapHnswGraph(config, initialCapacity);
    this.levelGenerator = new LevelGenerator(config);
    this.visitedSet = new EpochVisitedSet(initialCapacity);
    this.nodeEvalBuffer = new float[dimension];
    this.nodeEvaluator =
        (a, b) -> {
          this.storage.copyVector(a, this.nodeEvalBuffer);
          return this.calculator.distance(
              this.storage.vectorSegment(),
              this.storage.getVectorOffset(b),
              this.storage.getMin(b),
              this.storage.getScale(b),
              this.nodeEvalBuffer);
        };
  }

  public OffHeapQuantizedHnswIndex(
      int dimension,
      DistanceMetric metric,
      HnswConfig config,
      OffHeapQuantizedVectorStorage storage,
      OffHeapHnswGraph graph,
      OffHeapQuantizedEuclideanDistance calculator) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (metric != DistanceMetric.EUCLIDEAN) {
      throw new UnsupportedOperationException(
          "OffHeapQuantizedHnswIndex currently supports EUCLIDEAN metric only, got: " + metric);
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
          return this.calculator.distance(
              this.storage.vectorSegment(),
              this.storage.getVectorOffset(b),
              this.storage.getMin(b),
              this.storage.getScale(b),
              this.nodeEvalBuffer);
        };
  }

  @Override
  public synchronized void insert(long id, float[] vector) {
    VectorUtils.checkDimension(vector, dimension);
    VectorUtils.checkFinite(vector);

    // 1. Store vector in off-heap storage
    int internalId = storage.add(id, vector);

    // 2. Generate random level
    int nodeLevel = levelGenerator.nextLevel();

    // 3. Register node in off-heap graph
    graph.addNode(internalId, nodeLevel);

    // 4. Ensure visitedSet capacity
    visitedSet.ensureCapacity(internalId + 1);

    // 5. First node initializes the graph
    if (graph.size() == 1) {
      return;
    }

    HnswGraph.DistanceToQuery distanceToNew = buildDistanceToNode(internalId);
    NeighborSelector.NodeDistanceEvaluator nodeEval = nodeEvaluator;

    int currentEntryPoint = graph.entryPointId();
    int currentMaxLevel = graph.maxLevel();

    // 6. Phase 1: Greedy routing through upper layers
    for (int layer = currentMaxLevel; layer > nodeLevel; layer--) {
      if (graph.layout().maxLevel(currentEntryPoint) >= layer) {
        currentEntryPoint = graph.greedyClosest(distanceToNew, currentEntryPoint, layer);
      }
    }

    // 7. Phase 2: searchLayer + connect from min(currentMaxLevel, nodeLevel) down to 0
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

  @FunctionalInterface
  public interface ExactDistanceEvaluator {
    float distance(int internalId, float[] query);
  }

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

    List<NeighborSelector.Candidate> candidates = exploreCandidates(query, effectiveEf);

    int resultCount = Math.min(actualK, candidates.size());
    List<SearchResult> results = new ArrayList<>(resultCount);
    for (int i = 0; i < resultCount; i++) {
      NeighborSelector.Candidate c = candidates.get(i);
      long externalId = storage.getExternalId(c.id());
      results.add(new SearchResult(externalId, c.distance()));
    }
    return results;
  }

  public List<SearchResult> searchKnnCandidates(float[] query, int efSearch) {
    if (efSearch <= 0) {
      throw new IllegalArgumentException("efSearch must be positive: " + efSearch);
    }
    VectorUtils.checkDimension(query, dimension);
    VectorUtils.checkFinite(query);

    if (storage.size() == 0) {
      return Collections.emptyList();
    }

    List<NeighborSelector.Candidate> candidates = exploreCandidates(query, efSearch);
    List<SearchResult> results = new ArrayList<>(candidates.size());
    for (NeighborSelector.Candidate c : candidates) {
      long externalId = storage.getExternalId(c.id());
      results.add(new SearchResult(externalId, c.distance()));
    }
    return results;
  }

  public List<SearchResult> searchKnnWithRerank(
      float[] query, int k, int efSearch, ExactDistanceEvaluator exactEvaluator) {
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    if (efSearch <= 0) {
      throw new IllegalArgumentException("efSearch must be positive: " + efSearch);
    }
    Objects.requireNonNull(exactEvaluator, "exactEvaluator must not be null");
    VectorUtils.checkDimension(query, dimension);
    VectorUtils.checkFinite(query);

    if (storage.size() == 0) {
      return Collections.emptyList();
    }

    int actualK = Math.min(k, storage.size());
    int effectiveEf = Math.max(efSearch, actualK);

    List<NeighborSelector.Candidate> candidates = exploreCandidates(query, effectiveEf);
    if (candidates.isEmpty()) {
      return Collections.emptyList();
    }

    List<SearchResult> reranked = new ArrayList<>(candidates.size());
    for (NeighborSelector.Candidate c : candidates) {
      float exactDist = exactEvaluator.distance(c.id(), query);
      long externalId = storage.getExternalId(c.id());
      reranked.add(new SearchResult(externalId, exactDist));
    }

    reranked.sort(Comparator.comparingDouble(SearchResult::distance));
    int returnCount = Math.min(actualK, reranked.size());
    return new ArrayList<>(reranked.subList(0, returnCount));
  }

  public List<SearchResult> searchKnnWithRerank(
      float[] query, int k, int efSearch, VectorStorage rawStorage) {
    Objects.requireNonNull(rawStorage, "rawStorage must not be null");
    float[] rawBuffer = rawStorage.vectorBuffer();
    int dim = this.dimension;
    return searchKnnWithRerank(
        query,
        k,
        efSearch,
        (internalId, q) -> {
          int offset = internalId * dim;
          float sum = 0.0f;
          for (int i = 0; i < dim; i++) {
            float diff = q[i] - rawBuffer[offset + i];
            sum += diff * diff;
          }
          return sum;
        });
  }

  private List<NeighborSelector.Candidate> exploreCandidates(float[] query, int effectiveEf) {
    HnswGraph.DistanceToQuery distanceToQuery = buildDistanceToQuery(query);

    int currentEntryPoint = graph.entryPointId();
    int currentMaxLevel = graph.maxLevel();

    // Phase 1: Greedy descent from max level down to layer 1
    for (int layer = currentMaxLevel; layer >= 1; layer--) {
      currentEntryPoint = graph.greedyClosest(distanceToQuery, currentEntryPoint, layer);
    }

    // Phase 2: searchLayer at layer 0 with effectiveEf
    visitedSet.nextEpoch();
    return graph.searchLayer(
        distanceToQuery, new int[] {currentEntryPoint}, effectiveEf, 0, visitedSet);
  }

  private HnswGraph.DistanceToQuery buildDistanceToQuery(float[] query) {
    MemorySegment segment = storage.vectorSegment();
    return internalId ->
        calculator.distance(
            segment,
            storage.getVectorOffset(internalId),
            storage.getMin(internalId),
            storage.getScale(internalId),
            query);
  }

  private HnswGraph.DistanceToQuery buildDistanceToNode(int targetNodeId) {
    float[] targetVec = storage.getVector(targetNodeId);
    return buildDistanceToQuery(targetVec);
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
  public com.nanovector.core.storage.VectorDataView vectorData() {
    throw new UnsupportedOperationException(
        "OffHeapQuantizedHnswIndex maintains contiguous off-heap quantized byte storage. Use storage() to access OffHeapQuantizedVectorStorage.");
  }

  public HnswConfig config() {
    return config;
  }

  public OffHeapQuantizedVectorStorage storage() {
    return storage;
  }

  public OffHeapHnswGraph graph() {
    return graph;
  }

  public OffHeapQuantizedEuclideanDistance calculator() {
    return calculator;
  }

  /** Total native memory allocated across storage and graph topology layout (bytes). */
  public long nativeAllocatedBytes() {
    return storage.nativeAllocatedBytes() + graph.layout().nativeAllocatedBytes();
  }

  @Override
  public void close() {
    storage.close();
    graph.close();
  }
}
