package com.nanovector.core.index;

import com.nanovector.core.distance.DistanceCalculator;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.EpochVisitedSet;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.hnsw.HnswNode;
import com.nanovector.core.hnsw.LevelGenerator;
import com.nanovector.core.hnsw.NeighborSelector;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.storage.VectorStorage;
import com.nanovector.core.util.VectorUtils;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor index, conforming to the
 * {@link VectorIndex} contract.
 *
 * <p>Orchestrates {@link VectorStorage} (vector data &amp; ID mapping), {@link HnswGraph}
 * (multi-layer graph structure &amp; traversal), and {@link DistanceCalculator} (distance
 * computation) to provide empirically sublinear vector similarity search.
 *
 * <p><b>Architecture:</b>
 *
 * <pre>
 *   HnswIndex (public API: insert, searchKnn)
 *       │
 *       ├── VectorStorage   (float[] buffer, externalId ↔ internalId)
 *       │
 *       └── HnswGraph       (nodes, edges, searchLayer, greedyClosest)
 *               ├── HnswNode
 *               ├── LevelGenerator
 *               ├── NeighborSelector
 *               └── EpochVisitedSet
 * </pre>
 */
public final class HnswIndex implements VectorIndex {

  private final int dimension;
  private final DistanceMetric metric;
  private final DistanceCalculator calculator;
  private final VectorStorage storage;
  private final HnswGraph graph;
  private final HnswConfig config;
  private final LevelGenerator levelGenerator;
  private final EpochVisitedSet visitedSet;
  private final NeighborSelector.NodeDistanceEvaluator nodeEvaluator;

  public HnswIndex(int dimension, DistanceMetric metric, HnswConfig config) {
    this(dimension, metric, config, 1024, true);
  }

  public HnswIndex(int dimension, DistanceMetric metric, HnswConfig config, boolean useSimd) {
    this(dimension, metric, config, 1024, useSimd);
  }

  public HnswIndex(int dimension, DistanceMetric metric, HnswConfig config, int initialCapacity) {
    this(dimension, metric, config, initialCapacity, true);
  }

  public HnswIndex(
      int dimension,
      DistanceMetric metric,
      HnswConfig config,
      int initialCapacity,
      boolean useSimd) {
    this(dimension, metric, config, initialCapacity, DistanceCalculator.create(metric, useSimd));
  }

  public HnswIndex(
      int dimension,
      DistanceMetric metric,
      HnswConfig config,
      int initialCapacity,
      DistanceCalculator calculator) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    this.dimension = dimension;
    this.metric = Objects.requireNonNull(metric, "Metric must not be null");
    this.config = Objects.requireNonNull(config, "HnswConfig must not be null");
    this.calculator = Objects.requireNonNull(calculator, "Calculator must not be null");
    this.storage = new VectorStorage(dimension, initialCapacity);
    this.graph = new HnswGraph(config);
    this.levelGenerator = new LevelGenerator(config);
    this.visitedSet = new EpochVisitedSet(initialCapacity);
    this.nodeEvaluator =
        (a, b) ->
            this.calculator.distance(
                this.storage.getVectorBuffer(),
                a * dimension,
                this.storage.getVectorBuffer(),
                b * dimension,
                dimension);
  }

  // ── VectorIndex contract ────────────────────────────────────────────

  @Override
  public synchronized void insert(long id, float[] vector) {
    VectorUtils.checkDimension(vector, dimension);
    VectorUtils.checkFinite(vector);

    float[] effectiveVector;
    if (metric == DistanceMetric.COSINE) {
      effectiveVector = VectorUtils.normalize(vector);
    } else {
      effectiveVector = vector;
    }

    // 1. Store vector → get internalId
    int internalId = storage.insert(id, effectiveVector);

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

    // Build evaluators that close over storage and calculator
    HnswGraph.DistanceToQuery distanceToNew = buildDistanceToNode(internalId);
    NeighborSelector.NodeDistanceEvaluator nodeEval = nodeEvaluator;

    int currentEntryPoint = graph.entryPointId();
    int currentMaxLevel = graph.maxLevel();

    // 6. Phase 1: Greedy descent from current max level down to nodeLevel + 1
    //    (upper layers where the new node does NOT participate)
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

      // Select neighbors using Algorithm 4 heuristic with fallback
      int maxDegree = graph.maxDegreeForLayer(layer);
      int[] selectedNeighbors =
          NeighborSelector.selectNeighbors(new ArrayList<>(candidates), maxDegree, nodeEval);

      // Connect the new node bidirectionally to selected neighbors
      for (int neighborId : selectedNeighbors) {
        graph.connect(internalId, neighborId, layer, nodeEval);
      }

      // Update entry point for next lower layer
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
   * parameter.
   *
   * @param query query vector
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

    float[] effectiveQuery;
    if (metric == DistanceMetric.COSINE) {
      effectiveQuery = VectorUtils.normalize(query);
    } else {
      effectiveQuery = query;
    }

    int actualK = Math.min(k, storage.size());
    int effectiveEf = Math.max(efSearch, actualK);

    HnswGraph.DistanceToQuery distanceToQuery = buildDistanceToQuery(effectiveQuery);

    int currentEntryPoint = graph.entryPointId();
    int currentMaxLevel = graph.maxLevel();

    // Phase 1: Greedy descent from max level down to layer 1
    for (int layer = currentMaxLevel; layer >= 1; layer--) {
      currentEntryPoint = graph.greedyClosest(distanceToQuery, currentEntryPoint, layer);
    }

    // Phase 2: searchLayer at layer 0 with efSearch
    visitedSet.nextEpoch();
    List<NeighborSelector.Candidate> candidates =
        graph.searchLayer(
            distanceToQuery, new int[] {currentEntryPoint}, effectiveEf, 0, visitedSet);

    // Extract top-K and convert internalId → externalId
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

  // ── Configuration accessors ─────────────────────────────────────────

  public HnswConfig config() {
    return config;
  }

  public HnswGraph graph() {
    return graph;
  }

  public DistanceCalculator calculator() {
    return calculator;
  }

  // ── Distance evaluator factories ────────────────────────────────────

  /**
   * Creates a distance-to-query function for a query vector. Computes distance directly from the
   * contiguous VectorStorage buffer (zero-copy).
   */
  private HnswGraph.DistanceToQuery buildDistanceToQuery(float[] queryVector) {
    float[] buffer = storage.getVectorBuffer();
    return internalId -> calculator.distance(buffer, internalId * dimension, queryVector);
  }

  /**
   * Creates a distance-to-node function for computing distance from a stored node to other nodes.
   * Evaluates distance directly from the contiguous buffer without intermediate array allocations.
   */
  private HnswGraph.DistanceToQuery buildDistanceToNode(int targetInternalId) {
    int targetOffset = targetInternalId * dimension;
    float[] buffer = storage.getVectorBuffer();
    return internalId ->
        calculator.distance(buffer, internalId * dimension, buffer, targetOffset, dimension);
  }

  /** Returns the reusable node-to-node distance evaluator. */
  NeighborSelector.NodeDistanceEvaluator nodeDistanceEvaluator() {
    return nodeEvaluator;
  }
}
