package com.nanovector.persistence.integration;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.persistence.reader.NvecReader;
import com.nanovector.persistence.writer.NvecWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * End-to-end integration tests verifying the full lifecycle of vector indexes across creation,
 * query, serialization, deserialization, post-restore mutation, and re-serialization.
 */
class EndToEndPersistenceIntegrationTest {

  @TempDir Path tempDir;

  @Test
  @DisplayName("Full lifecycle for FlatIndex: build -> query -> save -> load -> mutate -> re-save")
  void testFlatIndexFullLifecycle() throws IOException {
    int dim = 16;
    FlatIndex original = new FlatIndex(dim, DistanceMetric.EUCLIDEAN);
    Random rand = new Random(111);

    for (long id = 1; id <= 100; id++) {
      float[] vec = new float[dim];
      for (int d = 0; d < dim; d++) vec[d] = rand.nextFloat();
      original.insert(id, vec);
    }

    float[] query = new float[dim];
    for (int d = 0; d < dim; d++) query[d] = 0.5f;
    List<SearchResult> expectedPreSave = original.searchKnn(query, 5);

    // 1. Serialize
    Path gen1Path = tempDir.resolve("flat_gen1.nvec");
    NvecWriter.write(original, gen1Path);

    // 2. Deserialize
    FlatIndex restored = NvecReader.readFlat(gen1Path);
    assertThat(restored.size()).isEqualTo(100);

    // 3. Verify identical query results
    List<SearchResult> restoredResults = restored.searchKnn(query, 5);
    assertThat(restoredResults).hasSameSizeAs(expectedPreSave);
    for (int i = 0; i < 5; i++) {
      assertThat(restoredResults.get(i).id()).isEqualTo(expectedPreSave.get(i).id());
      assertThat(restoredResults.get(i).distance())
          .isCloseTo(expectedPreSave.get(i).distance(), org.assertj.core.data.Offset.offset(1e-5f));
    }

    // 4. Mutate restored index (post-restore insertions)
    float[] newVector = new float[dim];
    for (int d = 0; d < dim; d++) newVector[d] = 0.5f; // Exact match to query
    restored.insert(9999L, newVector);
    assertThat(restored.size()).isEqualTo(101);

    // 5. Query again: newVector must now be rank #1 with distance ~0.0
    List<SearchResult> postMutationResults = restored.searchKnn(query, 1);
    assertThat(postMutationResults.get(0).id()).isEqualTo(9999L);
    assertThat(postMutationResults.get(0).distance())
        .isCloseTo(0.0f, org.assertj.core.data.Offset.offset(1e-5f));

    // 6. Re-serialize generation 2
    Path gen2Path = tempDir.resolve("flat_gen2.nvec");
    NvecWriter.write(restored, gen2Path);

    // 7. Verify generation 2 loads with 101 vectors
    FlatIndex gen2Restored = NvecReader.readFlat(gen2Path);
    assertThat(gen2Restored.size()).isEqualTo(101);
    assertThat(gen2Restored.searchKnn(query, 1).get(0).id()).isEqualTo(9999L);
  }

  @Test
  @DisplayName("Full lifecycle for HnswIndex: build -> query -> save -> load -> mutate -> re-save")
  void testHnswIndexFullLifecycle() throws IOException {
    int dim = 16;
    HnswConfig config = HnswConfig.withSeed(222L).withEfSearch(40);
    HnswIndex original = new HnswIndex(dim, DistanceMetric.EUCLIDEAN, config);
    Random rand = new Random(222);

    for (long id = 1; id <= 150; id++) {
      float[] vec = new float[dim];
      for (int d = 0; d < dim; d++) vec[d] = rand.nextFloat();
      original.insert(id, vec);
    }

    float[] query = new float[dim];
    for (int d = 0; d < dim; d++) query[d] = 0.5f;
    List<SearchResult> expectedPreSave = original.searchKnn(query, 5);

    // 1. Serialize
    Path gen1Path = tempDir.resolve("hnsw_gen1.nvec");
    NvecWriter.write(original, gen1Path);

    // 2. Deserialize
    HnswIndex restored = NvecReader.readHnsw(gen1Path);
    assertThat(restored.size()).isEqualTo(150);

    // 3. Verify identical query results
    List<SearchResult> restoredResults = restored.searchKnn(query, 5);
    assertThat(restoredResults).hasSameSizeAs(expectedPreSave);
    for (int i = 0; i < 5; i++) {
      assertThat(restoredResults.get(i).id()).isEqualTo(expectedPreSave.get(i).id());
      assertThat(restoredResults.get(i).distance())
          .isCloseTo(expectedPreSave.get(i).distance(), org.assertj.core.data.Offset.offset(1e-5f));
    }

    // 4. Mutate restored HNSW index (insert new node, connects into verbatim graph topology)
    float[] newVector = new float[dim];
    for (int d = 0; d < dim; d++) newVector[d] = 0.5f;
    restored.insert(8888L, newVector);
    assertThat(restored.size()).isEqualTo(151);

    // 5. Query: node 8888 should be at the top
    List<SearchResult> postMutationResults = restored.searchKnn(query, 1);
    assertThat(postMutationResults.get(0).id()).isEqualTo(8888L);
    assertThat(postMutationResults.get(0).distance())
        .isCloseTo(0.0f, org.assertj.core.data.Offset.offset(1e-5f));

    // 6. Re-serialize generation 2
    Path gen2Path = tempDir.resolve("hnsw_gen2.nvec");
    NvecWriter.write(restored, gen2Path);

    // 7. Verify generation 2 loads correctly with updated topology
    HnswIndex gen2Restored = NvecReader.readHnsw(gen2Path);
    assertThat(gen2Restored.size()).isEqualTo(151);
    assertThat(gen2Restored.searchKnn(query, 1).get(0).id()).isEqualTo(8888L);
  }
}
