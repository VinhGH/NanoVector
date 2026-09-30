package com.nanovector.core.offheap;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.QuantizedHnswIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.core.storage.VectorStorage;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class OffHeapQuantizedHnswIndexTest {

  @Test
  @DisplayName("Should insert vectors and return valid k-NN search results from off-heap index")
  void shouldInsertAndSearchKnn() {
    int dim = 64;
    int n = 150;
    long seed = 42L;

    HnswConfig config = new HnswConfig(16, 32, 100, 50, 1.0 / Math.log(16), seed);

    try (OffHeapQuantizedHnswIndex index =
        new OffHeapQuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, config, 128, true)) {
      Random random = new Random(seed);
      for (int i = 0; i < n; i++) {
        float[] vec = new float[dim];
        for (int d = 0; d < dim; d++) {
          vec[d] = random.nextFloat() * 2.0f - 1.0f;
        }
        index.insert(1000L + i, vec);
      }

      assertThat(index.size()).isEqualTo(n);
      assertThat(index.nativeAllocatedBytes()).isGreaterThan(0L);

      // Search with query vector identical to index vector 0 (should return externalId 1000 at rank
      // 0)
      float[] query = index.storage().getVector(0);
      List<SearchResult> results = index.searchKnn(query, 5);

      assertThat(results).hasSize(5);
      assertThat(results.get(0).id()).isEqualTo(1000L);
      assertThat(results.get(0).distance()).isLessThan(0.05f); // SQ8 approximation
    }
  }

  @Test
  @DisplayName(
      "Should exhibit behavioral equivalence with on-heap QuantizedHnswIndex under identical data and seed")
  void shouldExhibitEquivalenceWithOnHeapQuantizedHnsw() {
    int dim = 32;
    int n = 200;
    long seed = 12345L;

    HnswConfig config = new HnswConfig(16, 32, 100, 50, 1.0 / Math.log(16), seed);

    QuantizedHnswIndex onHeapIndex =
        new QuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, config, n, true);

    try (OffHeapQuantizedHnswIndex offHeapIndex =
        new OffHeapQuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, config, n, true)) {

      Random random = new Random(seed);
      float[][] vectors = new float[n][dim];
      for (int i = 0; i < n; i++) {
        for (int d = 0; d < dim; d++) {
          vectors[i][d] = random.nextFloat() * 2.0f - 1.0f;
        }
        long extId = 100L + i;
        onHeapIndex.insert(extId, vectors[i]);
        offHeapIndex.insert(extId, vectors[i]);
      }

      assertThat(offHeapIndex.size()).isEqualTo(onHeapIndex.size());
      assertThat(offHeapIndex.graph().entryPointId()).isEqualTo(onHeapIndex.graph().entryPointId());
      assertThat(offHeapIndex.graph().maxLevel()).isEqualTo(onHeapIndex.graph().maxLevel());

      // Query across 20 test vectors
      Random qRand = new Random(9999L);
      for (int q = 0; q < 20; q++) {
        float[] query = new float[dim];
        for (int d = 0; d < dim; d++) {
          query[d] = qRand.nextFloat() * 2.0f - 1.0f;
        }

        List<SearchResult> onHeapResults = onHeapIndex.searchKnn(query, 10, 50);
        List<SearchResult> offHeapResults = offHeapIndex.searchKnn(query, 10, 50);

        assertThat(offHeapResults).hasSameSizeAs(onHeapResults);
        for (int k = 0; k < onHeapResults.size(); k++) {
          assertThat(offHeapResults.get(k).id())
              .as("Mismatch at rank %d for query %d", k, q)
              .isEqualTo(onHeapResults.get(k).id());

          assertThat(offHeapResults.get(k).distance())
              .as("Distance mismatch at rank %d for query %d", k, q)
              .isCloseTo(
                  onHeapResults.get(k).distance(), org.assertj.core.data.Offset.offset(1e-4f));
        }
      }
    }
  }

  @Test
  @DisplayName("Should execute two-phase re-ranking against FP32 VectorStorage correctly")
  void shouldExecuteTwoPhaseRerank() {
    int dim = 16;
    int n = 100;
    long seed = 777L;

    HnswConfig config = new HnswConfig(8, 16, 50, 30, 1.0 / Math.log(8), seed);
    VectorStorage rawStorage = new VectorStorage(dim, n);

    try (OffHeapQuantizedHnswIndex offHeapIndex =
        new OffHeapQuantizedHnswIndex(dim, DistanceMetric.EUCLIDEAN, config, n, true)) {

      Random random = new Random(seed);
      for (int i = 0; i < n; i++) {
        float[] vec = new float[dim];
        for (int d = 0; d < dim; d++) {
          vec[d] = random.nextFloat();
        }
        long extId = 5000L + i;
        rawStorage.insert(extId, vec);
        offHeapIndex.insert(extId, vec);
      }

      float[] query = rawStorage.getVector(0);
      List<SearchResult> reranked = offHeapIndex.searchKnnWithRerank(query, 5, 30, rawStorage);

      assertThat(reranked).hasSize(5);
      // Vector 0 is the exact query vector, so distance should be 0.0f
      assertThat(reranked.get(0).id()).isEqualTo(5000L);
      assertThat(reranked.get(0).distance())
          .isCloseTo(0.0f, org.assertj.core.data.Offset.offset(1e-6f));
    }
  }

  @Test
  @DisplayName("Should release native resources on close() and throw on subsequent access")
  void shouldReleaseResourcesOnClose() {
    HnswConfig config = new HnswConfig(4, 8, 20, 10, 1.0 / Math.log(4), null);
    OffHeapQuantizedHnswIndex index =
        new OffHeapQuantizedHnswIndex(8, DistanceMetric.EUCLIDEAN, config, 16, false);
    index.insert(1L, new float[8]);
    index.close();

    assertThatThrownBy(() -> index.insert(2L, new float[8]))
        .isInstanceOf(IllegalStateException.class);
  }
}
