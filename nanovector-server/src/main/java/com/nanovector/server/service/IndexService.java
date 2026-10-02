package com.nanovector.server.service;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.QuantizedHnswIndex;
import com.nanovector.core.index.VectorIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.offheap.OffHeapQuantizedHnswIndex;
import com.nanovector.persistence.exception.CorruptIndexException;
import com.nanovector.persistence.reader.NvecReader;
import com.nanovector.persistence.writer.NvecWriter;
import com.nanovector.server.concurrency.ConcurrentSearchSession;
import com.nanovector.server.configuration.ServerProperties;
import com.nanovector.server.dto.CreateIndexRequest;
import com.nanovector.server.dto.IndexSummaryResponse;
import com.nanovector.server.dto.InsertVectorsRequest;
import com.nanovector.server.dto.InsertVectorsResponse;
import com.nanovector.server.dto.LoadIndexRequest;
import com.nanovector.server.dto.QueryRequest;
import com.nanovector.server.dto.QueryResponse;
import com.nanovector.server.dto.SaveIndexRequest;
import com.nanovector.server.dto.SaveIndexResponse;
import com.nanovector.server.dto.SearchResultItem;
import com.nanovector.server.dto.VectorItem;
import com.nanovector.server.exception.IndexAlreadyExistsException;
import com.nanovector.server.exception.IndexNotFoundException;
import com.nanovector.server.exception.PayloadTooLargeException;
import com.nanovector.server.lifecycle.IndexLifecycleManager;
import com.nanovector.server.model.IndexMetadata;
import com.nanovector.server.model.ServerIndexType;
import com.nanovector.server.registry.DefaultManagedIndex;
import com.nanovector.server.registry.IndexRegistry;
import com.nanovector.server.registry.ManagedIndex;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;
import org.springframework.stereotype.Service;

/**
 * Core business service coordinating vector index lifecycles, concurrency locks, persistence, and
 * similarity search operations.
 */
@Service
public class IndexService {

  private static final Pattern INDEX_NAME_PATTERN = Pattern.compile("^[a-zA-Z0-9_-]{1,64}$");

  private final IndexRegistry registry;
  private final ConcurrentSearchSession searchSession;
  private final StoragePathResolver pathResolver;
  private final ServerProperties serverProperties;

  public IndexService(
      IndexRegistry registry,
      ConcurrentSearchSession searchSession,
      StoragePathResolver pathResolver,
      ServerProperties serverProperties) {
    this.registry = Objects.requireNonNull(registry, "registry must not be null");
    this.searchSession = Objects.requireNonNull(searchSession, "searchSession must not be null");
    this.pathResolver = Objects.requireNonNull(pathResolver, "pathResolver must not be null");
    this.serverProperties =
        Objects.requireNonNull(serverProperties, "serverProperties must not be null");
  }

  /** Creates and registers a new vector index. */
  public IndexSummaryResponse createIndex(CreateIndexRequest request) {
    if (request == null) {
      throw new IllegalArgumentException("Create index request must not be null");
    }
    validateIndexName(request.name());
    if (registry.contains(request.name())) {
      throw new IndexAlreadyExistsException("Index already exists: " + request.name());
    }

    if (request.dimension() == null || request.dimension() <= 0) {
      throw new IllegalArgumentException("Dimension must be positive, got: " + request.dimension());
    }
    if (request.dimension() > 4096) {
      throw new IllegalArgumentException(
          "Dimension exceeds maximum supported limit (4096): " + request.dimension());
    }

    DistanceMetric metric = parseMetric(request.metric());
    ServerIndexType indexType = ServerIndexType.parse(request.type());

    Map<String, Object> params = request.parameters() != null ? request.parameters() : Map.of();
    Map<String, Object> extraMetadata = new HashMap<>(params);

    ManagedIndex managedIndex;
    try {
      managedIndex =
          IndexLifecycleManager.safelyCreate(
              request.name(),
              new IndexMetadata(
                  request.name(), indexType.name(), request.dimension(), metric, 0, extraMetadata),
              () -> buildIndex(indexType, request.dimension(), metric, params));
    } catch (Exception e) {
      if (e instanceof IllegalArgumentException iae) {
        throw iae;
      }
      throw new IllegalArgumentException("Failed to construct index: " + e.getMessage(), e);
    }

    registry.register(managedIndex);
    return toSummary(managedIndex);
  }

  /** Lists all registered vector indices. */
  public List<IndexSummaryResponse> listIndexes() {
    List<ManagedIndex> indices = registry.list();
    List<IndexSummaryResponse> result = new ArrayList<>(indices.size());
    for (ManagedIndex managed : indices) {
      result.add(managed.executeRead(idx -> toSummary(managed)));
    }
    return result;
  }

  /** Retrieves metadata for a specific index. */
  public IndexSummaryResponse getIndex(String name) {
    ManagedIndex managed = getManagedRequired(name);
    return managed.executeRead(idx -> toSummary(managed));
  }

  /** Deletes and safely disposes of an index. */
  public void deleteIndex(String name) {
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("Index name must not be blank");
    }
    boolean removed = registry.delete(name);
    if (!removed) {
      throw new IndexNotFoundException("Index not found: " + name);
    }
  }

  /** Inserts a batch of vectors under exclusive write lock protection. */
  public InsertVectorsResponse insertVectors(String name, InsertVectorsRequest request) {
    ManagedIndex managed = getManagedRequired(name);
    if (request == null || request.vectors() == null || request.vectors().isEmpty()) {
      throw new IllegalArgumentException("Vector batch must not be null or empty");
    }

    int batchSize = request.vectors().size();
    if (batchSize > serverProperties.getMaxBatchSize()) {
      throw new PayloadTooLargeException(
          "Batch size "
              + batchSize
              + " exceeds maximum allowed payload limit "
              + serverProperties.getMaxBatchSize());
    }

    int expectedDim = managed.metadata().dimension();
    Set<Long> seenBatchIds = new HashSet<>(batchSize);

    // Pre-validate entire batch prior to acquiring write lock or mutating data
    for (VectorItem item : request.vectors()) {
      if (item.id() == null) {
        throw new IllegalArgumentException("Vector item id must not be null");
      }
      if (!seenBatchIds.add(item.id())) {
        throw new IllegalArgumentException("Duplicate vector id in batch payload: " + item.id());
      }
      if (item.values() == null) {
        throw new IllegalArgumentException(
            "Vector values array must not be null for id: " + item.id());
      }
      if (item.values().length != expectedDim) {
        throw new IllegalArgumentException(
            "Vector dimension mismatch for id "
                + item.id()
                + ": expected "
                + expectedDim
                + ", got "
                + item.values().length);
      }
      for (float val : item.values()) {
        if (Float.isNaN(val) || Float.isInfinite(val)) {
          throw new IllegalArgumentException(
              "Vector contains NaN or Infinite values for id: " + item.id());
        }
      }
    }

    // Mutate index exclusively holding the write lock
    return managed.executeWrite(
        idx -> {
          for (VectorItem item : request.vectors()) {
            idx.insert(item.id(), item.values());
          }
          return new InsertVectorsResponse(batchSize, idx.size());
        });
  }

  /** Executes k-NN similarity query under shared read lock with isolated visited context. */
  public QueryResponse query(String name, QueryRequest request) {
    ManagedIndex managed = getManagedRequired(name);
    if (request == null || request.vector() == null) {
      throw new IllegalArgumentException("Query vector must not be null");
    }

    int expectedDim = managed.metadata().dimension();
    if (request.vector().length != expectedDim) {
      throw new IllegalArgumentException(
          "Query vector dimension mismatch: expected "
              + expectedDim
              + ", got "
              + request.vector().length);
    }
    for (float val : request.vector()) {
      if (Float.isNaN(val) || Float.isInfinite(val)) {
        throw new IllegalArgumentException("Query vector contains NaN or Infinite value");
      }
    }

    int k = request.k() != null ? request.k() : 10;
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive, got: " + k);
    }

    int efSearch = request.efSearch() != null ? request.efSearch() : Math.max(k, 16);
    if (efSearch <= 0) {
      throw new IllegalArgumentException("efSearch must be positive, got: " + efSearch);
    }

    long startNs = System.nanoTime();
    List<SearchResult> results = searchSession.search(managed, request.vector(), k, efSearch);
    long durationMicros = (System.nanoTime() - startNs) / 1000;

    List<SearchResultItem> items = new ArrayList<>(results.size());
    for (SearchResult sr : results) {
      items.add(new SearchResultItem(sr.id(), sr.distance()));
    }

    return new QueryResponse(name, k, items, durationMicros);
  }

  /** Persists an index holding the read lock throughout the entire serialization duration. */
  public SaveIndexResponse saveIndex(String name, SaveIndexRequest request) {
    ManagedIndex managed = getManagedRequired(name);

    ServerIndexType indexType = ServerIndexType.parse(managed.metadata().indexType());
    if (!indexType.isPersistenceSupported()) {
      throw new IllegalArgumentException(
          "Persistence is currently only supported for FLAT and HNSW indexes in NVEC v1, got: "
              + managed.metadata().indexType());
    }

    String fileName =
        (request != null && request.fileName() != null && !request.fileName().isBlank())
            ? request.fileName()
            : name + ".nvec";

    Path targetPath = pathResolver.resolveSafe(fileName);
    try {
      pathResolver.ensureBaseDirExists();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }

    // Execute serialization strictly inside read lock to prevent concurrent mutation tearing
    managed.executeRead(
        idx -> {
          try {
            NvecWriter.write(idx, targetPath);
            return null;
          } catch (IOException e) {
            throw new UncheckedIOException(e);
          }
        });

    long fileSizeBytes;
    try {
      fileSizeBytes = Files.size(targetPath);
    } catch (IOException e) {
      fileSizeBytes = -1;
    }

    return new SaveIndexResponse(
        name, targetPath.getFileName().toString(), managed.metadata().size(), fileSizeBytes);
  }

  /** Loads an index from the server data directory into the registry. */
  public IndexSummaryResponse loadIndex(LoadIndexRequest request) {
    if (request == null || request.name() == null || request.name().isBlank()) {
      throw new IllegalArgumentException("Index name must not be null or blank");
    }
    validateIndexName(request.name());
    if (registry.contains(request.name())) {
      throw new IndexAlreadyExistsException("Index already exists: " + request.name());
    }

    String fileName =
        (request.fileName() != null && !request.fileName().isBlank())
            ? request.fileName()
            : request.name() + ".nvec";

    Path targetPath = pathResolver.resolveSafe(fileName);
    if (!Files.exists(targetPath)) {
      throw new NoSuchElementException(
          "Persisted index file not found: " + targetPath.getFileName());
    }

    VectorIndex restoredIndex = null;
    ManagedIndex managedIndex;
    try {
      restoredIndex = NvecReader.read(targetPath);
      IndexMetadata metadata = extractMetadata(request.name(), restoredIndex);
      managedIndex = new DefaultManagedIndex(request.name(), restoredIndex, metadata);
    } catch (Throwable t) {
      if (restoredIndex instanceof AutoCloseable autoCloseable) {
        try {
          autoCloseable.close();
        } catch (Exception closeEx) {
          t.addSuppressed(closeEx);
        }
      }
      if (t instanceof CorruptIndexException cie) {
        throw new IllegalArgumentException("Corrupt index file: " + cie.getMessage(), cie);
      } else if (t instanceof RuntimeException re) {
        throw re;
      } else {
        throw new IllegalArgumentException("Failed to read index file: " + t.getMessage(), t);
      }
    }

    registry.register(managedIndex);
    return toSummary(managedIndex);
  }

  private ManagedIndex getManagedRequired(String name) {
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("Index name must not be blank");
    }
    return registry
        .get(name)
        .orElseThrow(() -> new IndexNotFoundException("Index not found: " + name));
  }

  private void validateIndexName(String name) {
    if (name == null || !INDEX_NAME_PATTERN.matcher(name).matches()) {
      throw new IllegalArgumentException(
          "Invalid index name '"
              + name
              + "'. Index names must be 1-64 characters and contain only letters, numbers, underscores, or hyphens.");
    }
  }

  private DistanceMetric parseMetric(String metricStr) {
    if (metricStr == null || metricStr.isBlank()) {
      return DistanceMetric.EUCLIDEAN;
    }
    try {
      return DistanceMetric.valueOf(metricStr.trim().toUpperCase(Locale.ROOT));
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException(
          "Unsupported distance metric '"
              + metricStr
              + "'. Supported metrics: EUCLIDEAN, COSINE, DOT_PRODUCT");
    }
  }

  private VectorIndex buildIndex(
      ServerIndexType type, int dimension, DistanceMetric metric, Map<String, Object> params) {
    return switch (type) {
      case FLAT -> new FlatIndex(dimension, metric);
      case HNSW -> {
        HnswConfig config = buildHnswConfig(params);
        yield new HnswIndex(dimension, metric, config);
      }
      case HNSW_SQ8 -> {
        if (metric != DistanceMetric.EUCLIDEAN) {
          throw new IllegalArgumentException(
              "HNSW_SQ8 index only supports EUCLIDEAN metric, got: " + metric);
        }
        HnswConfig config = buildHnswConfig(params);
        yield new QuantizedHnswIndex(dimension, metric, config);
      }
      case HNSW_SQ8_OFFHEAP -> {
        if (metric != DistanceMetric.EUCLIDEAN) {
          throw new IllegalArgumentException(
              "HNSW_SQ8_OFFHEAP index only supports EUCLIDEAN metric, got: " + metric);
        }
        HnswConfig config = buildHnswConfig(params);
        yield new OffHeapQuantizedHnswIndex(dimension, metric, config);
      }
    };
  }

  private HnswConfig buildHnswConfig(Map<String, Object> params) {
    int m = getIntParam(params, "m", 16);
    int m0 = getIntParam(params, "m0", 2 * m);
    int efConstruction = getIntParam(params, "efConstruction", 200);
    int efSearch = getIntParam(params, "efSearch", 50);

    if (m < 2) {
      throw new IllegalArgumentException("HNSW parameter 'm' must be at least 2, got: " + m);
    }
    if (m0 < m) {
      throw new IllegalArgumentException(
          "HNSW parameter 'm0' must be >= m (" + m + "), got: " + m0);
    }
    if (efConstruction <= 0) {
      throw new IllegalArgumentException(
          "HNSW parameter 'efConstruction' must be positive, got: " + efConstruction);
    }
    if (efSearch <= 0) {
      throw new IllegalArgumentException(
          "HNSW parameter 'efSearch' must be positive, got: " + efSearch);
    }

    return new HnswConfig(m, m0, efConstruction, efSearch, 1.0 / Math.log(m), null);
  }

  private int getIntParam(Map<String, Object> params, String key, int defaultValue) {
    if (params == null || !params.containsKey(key)) {
      return defaultValue;
    }
    Object val = params.get(key);
    if (val instanceof Number num) {
      return num.intValue();
    }
    try {
      return Integer.parseInt(val.toString().trim());
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException("Invalid integer parameter '" + key + "': " + val);
    }
  }

  private IndexMetadata extractMetadata(String name, VectorIndex index) {
    String type =
        (index instanceof FlatIndex)
            ? "FLAT"
            : (index instanceof HnswIndex)
                ? "HNSW"
                : (index instanceof QuantizedHnswIndex)
                    ? "HNSW_SQ8"
                    : (index instanceof OffHeapQuantizedHnswIndex) ? "HNSW_SQ8_OFFHEAP" : "UNKNOWN";

    Map<String, Object> extra = new HashMap<>();
    if (index instanceof HnswIndex hnsw) {
      extra.put("m", hnsw.config().m());
      extra.put("m0", hnsw.config().m0());
      extra.put("efConstruction", hnsw.config().efConstruction());
      extra.put("efSearch", hnsw.config().efSearch());
    }
    return new IndexMetadata(name, type, index.dimension(), index.metric(), index.size(), extra);
  }

  private IndexSummaryResponse toSummary(ManagedIndex managed) {
    IndexMetadata meta = managed.metadata();
    return new IndexSummaryResponse(
        meta.name(),
        meta.indexType(),
        meta.dimension(),
        meta.metric().name(),
        meta.size(),
        meta.createdAt(),
        meta.lastModifiedAt(),
        meta.extraProperties());
  }
}
