package com.nanovector.server.controller;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nanovector.server.NanoVectorServerApplication;
import com.nanovector.server.dto.CreateIndexRequest;
import com.nanovector.server.dto.InsertVectorsRequest;
import com.nanovector.server.dto.LoadIndexRequest;
import com.nanovector.server.dto.QueryRequest;
import com.nanovector.server.dto.SaveIndexRequest;
import com.nanovector.server.dto.VectorItem;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest(classes = NanoVectorServerApplication.class)
@AutoConfigureMockMvc
@DisplayName("IndexController Comprehensive REST API Quality Gate Tests")
class IndexControllerTest {

  static {
    System.setProperty("spring.classformat.ignore", "true");
  }

  static final Path TEST_DATA_DIR;

  static {
    try {
      TEST_DATA_DIR = Files.createTempDirectory("nanovector-rest-test-data");
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  @DynamicPropertySource
  static void configureProperties(DynamicPropertyRegistry registry) {
    registry.add("nanovector.server.data-dir", () -> TEST_DATA_DIR.toAbsolutePath().toString());
    registry.add("nanovector.server.max-batch-size", () -> "50");
  }

  @Autowired private MockMvc mockMvc;

  @Autowired private ObjectMapper objectMapper;

  @Test
  @DisplayName("1. Create FLAT and HNSW indexes successfully (201 Created)")
  void testCreateFlatAndHnswSuccess() throws Exception {
    CreateIndexRequest flatReq =
        new CreateIndexRequest("test_flat_idx", "FLAT", 4, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(flatReq)))
        .andExpect(status().isCreated())
        .andExpect(jsonPath("$.name").value("test_flat_idx"))
        .andExpect(jsonPath("$.type").value("FLAT"))
        .andExpect(jsonPath("$.dimension").value(4))
        .andExpect(jsonPath("$.metric").value("EUCLIDEAN"))
        .andExpect(jsonPath("$.size").value(0));

    CreateIndexRequest hnswReq =
        new CreateIndexRequest(
            "test_hnsw_idx",
            "HNSW",
            4,
            "COSINE",
            Map.of("m", 16, "efConstruction", 100, "efSearch", 30));
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(hnswReq)))
        .andExpect(status().isCreated())
        .andExpect(jsonPath("$.name").value("test_hnsw_idx"))
        .andExpect(jsonPath("$.type").value("HNSW"))
        .andExpect(jsonPath("$.metric").value("COSINE"))
        .andExpect(jsonPath("$.parameters.m").value(16))
        .andExpect(jsonPath("$.parameters.efConstruction").value(100));

    // Also verify HNSW_SQ8 and HNSW_SQ8_OFFHEAP
    CreateIndexRequest sq8Req =
        new CreateIndexRequest("test_sq8_idx", "HNSW_SQ8", 4, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(sq8Req)))
        .andExpect(status().isCreated())
        .andExpect(jsonPath("$.name").value("test_sq8_idx"))
        .andExpect(jsonPath("$.type").value("HNSW_SQ8"));

    CreateIndexRequest offHeapReq =
        new CreateIndexRequest("test_offheap_idx", "HNSW_SQ8_OFFHEAP", 4, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(offHeapReq)))
        .andExpect(status().isCreated())
        .andExpect(jsonPath("$.name").value("test_offheap_idx"))
        .andExpect(jsonPath("$.type").value("HNSW_SQ8_OFFHEAP"));
  }

  @Test
  @DisplayName("2. Create duplicate index name returns 409 Conflict")
  void testCreateDuplicateNameReturns409() throws Exception {
    CreateIndexRequest req = new CreateIndexRequest("dup_name_idx", "FLAT", 4, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(req)))
        .andExpect(status().isCreated());

    // Attempt second creation with duplicate name
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(req)))
        .andExpect(status().isConflict())
        .andExpect(jsonPath("$.status").value(409))
        .andExpect(jsonPath("$.error").value("Conflict"))
        .andExpect(jsonPath("$.message").value("Index already exists: dup_name_idx"))
        .andExpect(jsonPath("$.path").value("/api/v1/indexes"));
  }

  @Test
  @DisplayName("3. Create index with invalid inputs returns 400 Bad Request")
  void testCreateInvalidInputsReturns400() throws Exception {
    // Blank name
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(
                    objectMapper.writeValueAsString(
                        new CreateIndexRequest("", "FLAT", 4, "EUCLIDEAN", null))))
        .andExpect(status().isBadRequest());

    // Name with path traversal
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(
                    objectMapper.writeValueAsString(
                        new CreateIndexRequest("../evil", "FLAT", 4, "EUCLIDEAN", null))))
        .andExpect(status().isBadRequest());

    // Non-positive dimension
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(
                    objectMapper.writeValueAsString(
                        new CreateIndexRequest("bad_dim", "FLAT", -1, "EUCLIDEAN", null))))
        .andExpect(status().isBadRequest());

    // Unknown index type
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(
                    objectMapper.writeValueAsString(
                        new CreateIndexRequest("bad_type", "UNKNOWN_TYPE", 4, "EUCLIDEAN", null))))
        .andExpect(status().isBadRequest());

    // SQ8 with non-Euclidean metric
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(
                    objectMapper.writeValueAsString(
                        new CreateIndexRequest("sq8_cosine", "HNSW_SQ8", 4, "COSINE", null))))
        .andExpect(status().isBadRequest());
  }

  @Test
  @DisplayName("4. List and Get Metadata return accurate information")
  void testListAndGetMetadata() throws Exception {
    CreateIndexRequest req =
        new CreateIndexRequest("meta_inspect_idx", "FLAT", 8, "DOT_PRODUCT", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(req)))
        .andExpect(status().isCreated());

    // Get specific index
    mockMvc
        .perform(get("/api/v1/indexes/meta_inspect_idx"))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.name").value("meta_inspect_idx"))
        .andExpect(jsonPath("$.dimension").value(8))
        .andExpect(jsonPath("$.metric").value("DOT_PRODUCT"))
        .andExpect(jsonPath("$.createdAt").isNotEmpty());

    // List all
    mockMvc
        .perform(get("/api/v1/indexes"))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$").isArray());
  }

  @Test
  @DisplayName("5. Delete index releases lifecycle properly (204 No Content)")
  void testDeleteIndexReleasesLifecycle() throws Exception {
    CreateIndexRequest req = new CreateIndexRequest("to_delete_idx", "FLAT", 4, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(req)))
        .andExpect(status().isCreated());

    // Delete
    mockMvc.perform(delete("/api/v1/indexes/to_delete_idx")).andExpect(status().isNoContent());

    // Verification: index should no longer exist
    mockMvc
        .perform(get("/api/v1/indexes/to_delete_idx"))
        .andExpect(status().isNotFound())
        .andExpect(jsonPath("$.status").value(404));

    // Second delete returns 404
    mockMvc.perform(delete("/api/v1/indexes/to_delete_idx")).andExpect(status().isNotFound());
  }

  @Test
  @DisplayName("6. Insert batch and query added vectors")
  void testInsertBatchAndQuery() throws Exception {
    CreateIndexRequest createReq =
        new CreateIndexRequest("batch_query_idx", "FLAT", 3, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(createReq)))
        .andExpect(status().isCreated());

    List<VectorItem> items =
        List.of(
            new VectorItem(101L, new float[] {1.0f, 0.0f, 0.0f}),
            new VectorItem(102L, new float[] {0.0f, 1.0f, 0.0f}),
            new VectorItem(103L, new float[] {0.0f, 0.0f, 1.0f}));
    InsertVectorsRequest insertReq = new InsertVectorsRequest(items);

    mockMvc
        .perform(
            post("/api/v1/indexes/batch_query_idx/vectors")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(insertReq)))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.insertedCount").value(3))
        .andExpect(jsonPath("$.totalSize").value(3));

    // Query for vector closest to [1.0, 0.1, 0.0]
    QueryRequest queryReq = new QueryRequest(new float[] {1.0f, 0.1f, 0.0f}, 2, null);
    mockMvc
        .perform(
            post("/api/v1/indexes/batch_query_idx/query")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(queryReq)))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.indexName").value("batch_query_idx"))
        .andExpect(jsonPath("$.k").value(2))
        .andExpect(jsonPath("$.results[0].id").value(101L));
  }

  @Test
  @DisplayName("7. Insert wrong dimension returns 400 and does NOT mutate data")
  void testInsertWrongDimensionDoesNotMutateData() throws Exception {
    CreateIndexRequest createReq =
        new CreateIndexRequest("atomic_dim_idx", "FLAT", 3, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(createReq)))
        .andExpect(status().isCreated());

    // Insert 1 valid vector initially
    mockMvc
        .perform(
            post("/api/v1/indexes/atomic_dim_idx/vectors")
                .contentType(MediaType.APPLICATION_JSON)
                .content(
                    objectMapper.writeValueAsString(
                        new InsertVectorsRequest(
                            List.of(new VectorItem(1L, new float[] {1.0f, 2.0f, 3.0f}))))))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.totalSize").value(1));

    // Attempt to insert batch where 2nd vector has dimension 2 instead of 3
    InsertVectorsRequest invalidBatch =
        new InsertVectorsRequest(
            List.of(
                new VectorItem(2L, new float[] {4.0f, 5.0f, 6.0f}),
                new VectorItem(3L, new float[] {7.0f, 8.0f}))); // dim 2!

    mockMvc
        .perform(
            post("/api/v1/indexes/atomic_dim_idx/vectors")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(invalidBatch)))
        .andExpect(status().isBadRequest())
        .andExpect(jsonPath("$.status").value(400))
        .andExpect(
            jsonPath("$.message")
                .value(org.hamcrest.Matchers.containsString("dimension mismatch")));

    // Verify index size remains strictly 1 (vector 2 was NOT inserted)
    mockMvc
        .perform(get("/api/v1/indexes/atomic_dim_idx"))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.size").value(1));
  }

  @Test
  @DisplayName("8. Duplicate ID returns 400 Bad Request")
  void testDuplicateIdReturns400() throws Exception {
    CreateIndexRequest createReq =
        new CreateIndexRequest("dup_id_idx", "FLAT", 2, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(createReq)))
        .andExpect(status().isCreated());

    // Duplicate within same batch
    InsertVectorsRequest sameBatchDup =
        new InsertVectorsRequest(
            List.of(
                new VectorItem(42L, new float[] {1.0f, 2.0f}),
                new VectorItem(42L, new float[] {3.0f, 4.0f})));

    mockMvc
        .perform(
            post("/api/v1/indexes/dup_id_idx/vectors")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(sameBatchDup)))
        .andExpect(status().isBadRequest())
        .andExpect(
            jsonPath("$.message")
                .value(org.hamcrest.Matchers.containsString("Duplicate vector id")));

    // Insert 42L successfully
    mockMvc
        .perform(
            post("/api/v1/indexes/dup_id_idx/vectors")
                .contentType(MediaType.APPLICATION_JSON)
                .content(
                    objectMapper.writeValueAsString(
                        new InsertVectorsRequest(
                            List.of(new VectorItem(42L, new float[] {1.0f, 2.0f}))))))
        .andExpect(status().isOk());

    // Attempt second insert with existing ID 42L
    mockMvc
        .perform(
            post("/api/v1/indexes/dup_id_idx/vectors")
                .contentType(MediaType.APPLICATION_JSON)
                .content(
                    objectMapper.writeValueAsString(
                        new InsertVectorsRequest(
                            List.of(new VectorItem(42L, new float[] {5.0f, 6.0f}))))))
        .andExpect(status().isBadRequest());
  }

  @Test
  @DisplayName("9. Query FLAT matches Ground Truth ranking exactly")
  void testQueryFlatMatchesGroundTruth() throws Exception {
    CreateIndexRequest createReq =
        new CreateIndexRequest("gt_flat_idx", "FLAT", 3, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(createReq)))
        .andExpect(status().isCreated());

    List<VectorItem> items =
        List.of(
            new VectorItem(10L, new float[] {1.0f, 0.0f, 0.0f}), // dist^2 = 1.00
            new VectorItem(20L, new float[] {2.0f, 0.0f, 0.0f}), // dist^2 = 4.00
            new VectorItem(30L, new float[] {3.0f, 0.0f, 0.0f}), // dist^2 = 9.00
            new VectorItem(40L, new float[] {0.5f, 0.0f, 0.0f}), // dist^2 = 0.25
            new VectorItem(50L, new float[] {0.1f, 0.0f, 0.0f})); // dist^2 = 0.01

    mockMvc
        .perform(
            post("/api/v1/indexes/gt_flat_idx/vectors")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(new InsertVectorsRequest(items))))
        .andExpect(status().isOk());

    // Query from origin [0, 0, 0]
    QueryRequest queryReq = new QueryRequest(new float[] {0.0f, 0.0f, 0.0f}, 3, null);

    mockMvc
        .perform(
            post("/api/v1/indexes/gt_flat_idx/query")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(queryReq)))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.results[0].id").value(50L))
        .andExpect(jsonPath("$.results[1].id").value(40L))
        .andExpect(jsonPath("$.results[2].id").value(10L));
  }

  @Test
  @DisplayName("10. Query HNSW supports efSearch parameter")
  void testQueryHnswSupportsEfSearch() throws Exception {
    CreateIndexRequest createReq =
        new CreateIndexRequest("ef_hnsw_idx", "HNSW", 4, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(createReq)))
        .andExpect(status().isCreated());

    List<VectorItem> items = new ArrayList<>();
    for (int i = 0; i < 20; i++) {
      items.add(new VectorItem((long) i, new float[] {i * 0.1f, 0.2f, 0.3f, 0.4f}));
    }
    mockMvc
        .perform(
            post("/api/v1/indexes/ef_hnsw_idx/vectors")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(new InsertVectorsRequest(items))))
        .andExpect(status().isOk());

    // Valid query with efSearch = 64
    QueryRequest validQuery = new QueryRequest(new float[] {0.0f, 0.2f, 0.3f, 0.4f}, 5, 64);
    mockMvc
        .perform(
            post("/api/v1/indexes/ef_hnsw_idx/query")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(validQuery)))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.results").isArray())
        .andExpect(jsonPath("$.results.length()").value(5));

    // Invalid efSearch <= 0 returns 400
    QueryRequest invalidQuery = new QueryRequest(new float[] {0.0f, 0.2f, 0.3f, 0.4f}, 5, -1);
    mockMvc
        .perform(
            post("/api/v1/indexes/ef_hnsw_idx/query")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(invalidQuery)))
        .andExpect(status().isBadRequest());
  }

  @Test
  @DisplayName("11. Query non-existent index returns 404")
  void testQueryNonExistentIndexReturns404() throws Exception {
    QueryRequest query = new QueryRequest(new float[] {1.0f, 2.0f}, 5, null);
    mockMvc
        .perform(
            post("/api/v1/indexes/non_existent_idx/query")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(query)))
        .andExpect(status().isNotFound())
        .andExpect(jsonPath("$.status").value(404));
  }

  @Test
  @DisplayName("12. Save then Load preserves data, metadata, and nearest-neighbor recall")
  void testSaveAndLoadPreservesDataAndMetadata() throws Exception {
    CreateIndexRequest createReq =
        new CreateIndexRequest(
            "persist_roundtrip_idx",
            "HNSW",
            3,
            "EUCLIDEAN",
            Map.of("m", 16, "efConstruction", 100));
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(createReq)))
        .andExpect(status().isCreated());

    List<VectorItem> items = new ArrayList<>();
    for (int i = 0; i < 15; i++) {
      items.add(new VectorItem((long) (i + 1), new float[] {i * 1.5f, (float) Math.sin(i), 0.5f}));
    }
    mockMvc
        .perform(
            post("/api/v1/indexes/persist_roundtrip_idx/vectors")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(new InsertVectorsRequest(items))))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.totalSize").value(15));

    // 1. Save to disk
    mockMvc
        .perform(
            post("/api/v1/indexes/persist_roundtrip_idx/save")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(new SaveIndexRequest("roundtrip.nvec"))))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.fileName").value("roundtrip.nvec"))
        .andExpect(jsonPath("$.size").value(15))
        .andExpect(jsonPath("$.fileSizeBytes").value(org.hamcrest.Matchers.greaterThan(0)));

    // 2. Delete original index
    mockMvc
        .perform(delete("/api/v1/indexes/persist_roundtrip_idx"))
        .andExpect(status().isNoContent());

    // 3. Load from disk into registry with name 'restored_roundtrip_idx'
    mockMvc
        .perform(
            post("/api/v1/indexes/load")
                .contentType(MediaType.APPLICATION_JSON)
                .content(
                    objectMapper.writeValueAsString(
                        new LoadIndexRequest("restored_roundtrip_idx", "roundtrip.nvec"))))
        .andExpect(status().isCreated())
        .andExpect(jsonPath("$.name").value("restored_roundtrip_idx"))
        .andExpect(jsonPath("$.type").value("HNSW"))
        .andExpect(jsonPath("$.dimension").value(3))
        .andExpect(jsonPath("$.size").value(15));

    // 4. Query loaded index
    mockMvc
        .perform(
            post("/api/v1/indexes/restored_roundtrip_idx/query")
                .contentType(MediaType.APPLICATION_JSON)
                .content(
                    objectMapper.writeValueAsString(
                        new QueryRequest(new float[] {0.0f, 0.0f, 0.5f}, 3, 32))))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.results.length()").value(3))
        .andExpect(jsonPath("$.results[0].id").value(1L)); // Closest to 0.0, 0.0, 0.5
  }

  @Test
  @DisplayName("13. Load faulty file does not leave registry entry or leak native memory")
  void testLoadFaultyFileDoesNotLeaveRegistryEntry() throws Exception {
    // Write corrupted header file
    Path corruptPath = TEST_DATA_DIR.resolve("corrupt_test.nvec");
    Files.write(corruptPath, new byte[] {0x00, 0x01, 0x02, 0x03, 0x04, 0x05});

    mockMvc
        .perform(
            post("/api/v1/indexes/load")
                .contentType(MediaType.APPLICATION_JSON)
                .content(
                    objectMapper.writeValueAsString(
                        new LoadIndexRequest("corrupt_idx", "corrupt_test.nvec"))))
        .andExpect(status().isBadRequest());

    // Ensure not in registry
    mockMvc.perform(get("/api/v1/indexes/corrupt_idx")).andExpect(status().isNotFound());
  }

  @Test
  @DisplayName("14. Batch vector payload exceeding maxBatchSize returns 413 Payload Too Large")
  void testPayloadTooLargeReturns413() throws Exception {
    CreateIndexRequest createReq =
        new CreateIndexRequest("payload_limit_idx", "FLAT", 2, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(createReq)))
        .andExpect(status().isCreated());

    // Configured max-batch-size is 50. Generate 55 items
    List<VectorItem> largeBatch = new ArrayList<>();
    for (int i = 0; i < 55; i++) {
      largeBatch.add(new VectorItem((long) i, new float[] {1.0f, 2.0f}));
    }

    mockMvc
        .perform(
            post("/api/v1/indexes/payload_limit_idx/vectors")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(new InsertVectorsRequest(largeBatch))))
        .andExpect(status().isPayloadTooLarge())
        .andExpect(jsonPath("$.status").value(413))
        .andExpect(jsonPath("$.error").value("Payload Too Large"))
        .andExpect(
            jsonPath("$.message")
                .value(
                    org.hamcrest.Matchers.containsString("exceeds maximum allowed payload limit")));
  }

  @Test
  @DisplayName("15. Attempting to save unsupported SQ8 index returns 400 Bad Request")
  void testSaveUnsupportedIndexReturns400() throws Exception {
    CreateIndexRequest sq8Req =
        new CreateIndexRequest("unsupported_save_idx", "HNSW_SQ8", 4, "EUCLIDEAN", null);
    mockMvc
        .perform(
            post("/api/v1/indexes")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(sq8Req)))
        .andExpect(status().isCreated());

    mockMvc
        .perform(
            post("/api/v1/indexes/unsupported_save_idx/save")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{}"))
        .andExpect(status().isBadRequest())
        .andExpect(jsonPath("$.status").value(400))
        .andExpect(
            jsonPath("$.message")
                .value(org.hamcrest.Matchers.containsString("supported for FLAT and HNSW")));
  }
}
