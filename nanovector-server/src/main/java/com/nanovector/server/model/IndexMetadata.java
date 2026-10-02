package com.nanovector.server.model;

import com.nanovector.core.distance.DistanceMetric;
import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Metadata descriptor for a vector index managed by NanoVector Server.
 *
 * <p>Tracks architectural properties (dimension, metric, storage engine), operational state (size,
 * timestamps), and engine-specific hyperparameters (e.g., M, efConstruction, efSearch).
 */
public final class IndexMetadata {

  private final String name;
  private final String indexType;
  private final int dimension;
  private final DistanceMetric metric;
  private final Instant createdAt;
  private final Map<String, Object> extraProperties;

  private volatile int size;
  private volatile Instant lastModifiedAt;

  public IndexMetadata(
      String name,
      String indexType,
      int dimension,
      DistanceMetric metric,
      int size,
      Map<String, Object> extraProperties) {
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("Index name must not be null or blank");
    }
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    this.name = name;
    this.indexType = Objects.requireNonNull(indexType, "indexType must not be null");
    this.dimension = dimension;
    this.metric = Objects.requireNonNull(metric, "metric must not be null");
    this.size = Math.max(0, size);
    this.createdAt = Instant.now();
    this.lastModifiedAt = this.createdAt;
    this.extraProperties =
        extraProperties == null
            ? Collections.emptyMap()
            : Collections.unmodifiableMap(new HashMap<>(extraProperties));
  }

  public static IndexMetadata of(
      String name, String indexType, int dimension, DistanceMetric metric, int size) {
    return new IndexMetadata(name, indexType, dimension, metric, size, Collections.emptyMap());
  }

  public String name() {
    return name;
  }

  public String indexType() {
    return indexType;
  }

  public int dimension() {
    return dimension;
  }

  public DistanceMetric metric() {
    return metric;
  }

  public int size() {
    return size;
  }

  public void updateSize(int newSize) {
    this.size = newSize;
  }

  public Instant createdAt() {
    return createdAt;
  }

  public Instant lastModifiedAt() {
    return lastModifiedAt;
  }

  public void updateLastModified() {
    this.lastModifiedAt = Instant.now();
  }

  public Map<String, Object> extraProperties() {
    return extraProperties;
  }
}
