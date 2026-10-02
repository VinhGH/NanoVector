package com.nanovector.server.registry;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.server.model.IndexMetadata;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("IndexRegistry Test Suite")
class IndexRegistryTest {

  @Test
  @DisplayName("Register and retrieve multiple indexes with distinct names")
  void registerAndRetrieveMultipleIndexes() {
    try (IndexRegistry registry = new IndexRegistry()) {
      ManagedIndex idx1 =
          new DefaultManagedIndex(
              "index-a",
              new FlatIndex(128, DistanceMetric.EUCLIDEAN),
              IndexMetadata.of("index-a", "FLAT", 128, DistanceMetric.EUCLIDEAN, 0));

      ManagedIndex idx2 =
          new DefaultManagedIndex(
              "index-b",
              new FlatIndex(64, DistanceMetric.COSINE),
              IndexMetadata.of("index-b", "FLAT", 64, DistanceMetric.COSINE, 0));

      registry.register(idx1);
      registry.register(idx2);

      assertThat(registry.size()).isEqualTo(2);
      assertThat(registry.contains("index-a")).isTrue();
      assertThat(registry.contains("index-b")).isTrue();
      assertThat(registry.contains("index-c")).isFalse();

      assertThat(registry.get("index-a")).contains(idx1);
      assertThat(registry.get("index-b")).contains(idx2);
      assertThat(registry.getRequired("index-a")).isSameAs(idx1);
      assertThat(registry.getRequired("index-b")).isSameAs(idx2);

      assertThat(registry.list()).containsExactlyInAnyOrder(idx1, idx2);
    }
  }

  @Test
  @DisplayName("Rejects duplicate index registration")
  void rejectsDuplicateRegistration() {
    try (IndexRegistry registry = new IndexRegistry()) {
      ManagedIndex idx1 =
          new DefaultManagedIndex(
              "same-name",
              new FlatIndex(128, DistanceMetric.EUCLIDEAN),
              IndexMetadata.of("same-name", "FLAT", 128, DistanceMetric.EUCLIDEAN, 0));

      ManagedIndex idx2 =
          new DefaultManagedIndex(
              "same-name",
              new FlatIndex(128, DistanceMetric.EUCLIDEAN),
              IndexMetadata.of("same-name", "FLAT", 128, DistanceMetric.EUCLIDEAN, 0));

      registry.register(idx1);

      assertThatThrownBy(() -> registry.register(idx2))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Index already exists: same-name");
    }
  }

  @Test
  @DisplayName("getRequired throws NoSuchElementException for missing index")
  void getRequiredThrowsOnMissingIndex() {
    try (IndexRegistry registry = new IndexRegistry()) {
      assertThatThrownBy(() -> registry.getRequired("non-existent"))
          .isInstanceOf(NoSuchElementException.class)
          .hasMessageContaining("Index not found: non-existent");
    }
  }

  @Test
  @DisplayName("Deleting index closes it and does not affect other registered indexes")
  void deleteIndexIsolation() {
    try (IndexRegistry registry = new IndexRegistry()) {
      ManagedIndex idx1 =
          new DefaultManagedIndex(
              "idx-keep",
              new FlatIndex(128, DistanceMetric.EUCLIDEAN),
              IndexMetadata.of("idx-keep", "FLAT", 128, DistanceMetric.EUCLIDEAN, 0));

      ManagedIndex idx2 =
          new DefaultManagedIndex(
              "idx-remove",
              new FlatIndex(128, DistanceMetric.EUCLIDEAN),
              IndexMetadata.of("idx-remove", "FLAT", 128, DistanceMetric.EUCLIDEAN, 0));

      registry.register(idx1);
      registry.register(idx2);

      boolean deleted = registry.delete("idx-remove");
      assertThat(deleted).isTrue();
      assertThat(idx2.isClosed()).isTrue();

      // idx1 must remain completely unaffected and open
      assertThat(registry.size()).isEqualTo(1);
      assertThat(registry.contains("idx-keep")).isTrue();
      assertThat(registry.contains("idx-remove")).isFalse();
      assertThat(idx1.isClosed()).isFalse();

      // Attempting to delete non-existent index returns false
      assertThat(registry.delete("idx-remove")).isFalse();
    }
  }

  @Test
  @DisplayName("Registry shutdown closes all registered indices")
  void shutdownClosesAllIndices() {
    AtomicInteger closeCounterA = new AtomicInteger(0);
    AtomicInteger closeCounterB = new AtomicInteger(0);

    class CloseTrackingIndex implements com.nanovector.core.index.VectorIndex, AutoCloseable {
      private final FlatIndex delegate = new FlatIndex(128, DistanceMetric.EUCLIDEAN);
      private final AtomicInteger counter;

      CloseTrackingIndex(AtomicInteger counter) {
        this.counter = counter;
      }

      @Override
      public void insert(long id, float[] vector) {
        delegate.insert(id, vector);
      }

      @Override
      public java.util.List<com.nanovector.core.model.SearchResult> searchKnn(
          float[] query, int k) {
        return delegate.searchKnn(query, k);
      }

      @Override
      public int size() {
        return delegate.size();
      }

      @Override
      public int dimension() {
        return delegate.dimension();
      }

      @Override
      public DistanceMetric metric() {
        return delegate.metric();
      }

      @Override
      public com.nanovector.core.storage.VectorDataView vectorData() {
        return delegate.vectorData();
      }

      @Override
      public void close() {
        counter.incrementAndGet();
      }
    }

    ManagedIndex idxA =
        new DefaultManagedIndex(
            "idx-a",
            new CloseTrackingIndex(closeCounterA),
            IndexMetadata.of("idx-a", "FLAT", 128, DistanceMetric.EUCLIDEAN, 0));
    ManagedIndex idxB =
        new DefaultManagedIndex(
            "idx-b",
            new CloseTrackingIndex(closeCounterB),
            IndexMetadata.of("idx-b", "FLAT", 128, DistanceMetric.EUCLIDEAN, 0));

    IndexRegistry registry = new IndexRegistry();
    registry.register(idxA);
    registry.register(idxB);

    registry.close();

    assertThat(registry.size()).isEqualTo(0);
    assertThat(idxA.isClosed()).isTrue();
    assertThat(idxB.isClosed()).isTrue();
    assertThat(closeCounterA.get()).isEqualTo(1);
    assertThat(closeCounterB.get()).isEqualTo(1);
  }
}
