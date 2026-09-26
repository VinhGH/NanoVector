package com.nanovector.core.index;

import com.nanovector.core.distance.DistanceCalculator;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.hnsw.HnswGraph;
import com.nanovector.core.storage.VectorStorage;
import java.util.Objects;

/**
 * Controlled internal bridge for restoring vector index instances directly from persistent storage
 * representations (.nvec) without re-indexing, re-connecting, or re-pruning graph edges.
 *
 * <p>Preserves encapsulation of {@link FlatIndex} and {@link HnswIndex} by invoking package-private
 * restoration constructors.
 */
public final class IndexRestorer {

  private IndexRestorer() {}

  /**
   * Restores a {@link FlatIndex} from pre-populated vector storage and distance calculator.
   *
   * @param dimension vector dimensionality
   * @param metric similarity distance metric
   * @param storage reconstructed vector storage
   * @param calculator distance calculator
   * @return restored {@link FlatIndex}
   */
  public static FlatIndex restoreFlat(
      int dimension, DistanceMetric metric, VectorStorage storage, DistanceCalculator calculator) {
    Objects.requireNonNull(metric, "DistanceMetric must not be null");
    Objects.requireNonNull(storage, "VectorStorage must not be null");
    Objects.requireNonNull(calculator, "DistanceCalculator must not be null");
    return new FlatIndex(dimension, metric, storage, calculator);
  }

  /**
   * Restores an {@link HnswIndex} from pre-populated storage, verbatim graph topology, and
   * configuration.
   *
   * @param dimension vector dimensionality
   * @param metric similarity distance metric
   * @param config HNSW hyperparameters
   * @param storage reconstructed vector storage
   * @param graph reconstructed graph topology
   * @param calculator distance calculator
   * @return restored {@link HnswIndex}
   */
  public static HnswIndex restoreHnsw(
      int dimension,
      DistanceMetric metric,
      HnswConfig config,
      VectorStorage storage,
      HnswGraph graph,
      DistanceCalculator calculator) {
    Objects.requireNonNull(metric, "DistanceMetric must not be null");
    Objects.requireNonNull(config, "HnswConfig must not be null");
    Objects.requireNonNull(storage, "VectorStorage must not be null");
    Objects.requireNonNull(graph, "HnswGraph must not be null");
    Objects.requireNonNull(calculator, "DistanceCalculator must not be null");
    return new HnswIndex(dimension, metric, config, storage, graph, calculator);
  }
}
