package com.nanovector.server.controller;

import com.nanovector.server.dto.CreateIndexRequest;
import com.nanovector.server.dto.IndexSummaryResponse;
import com.nanovector.server.dto.InsertVectorsRequest;
import com.nanovector.server.dto.InsertVectorsResponse;
import com.nanovector.server.dto.LoadIndexRequest;
import com.nanovector.server.dto.QueryRequest;
import com.nanovector.server.dto.QueryResponse;
import com.nanovector.server.dto.SaveIndexRequest;
import com.nanovector.server.dto.SaveIndexResponse;
import com.nanovector.server.service.IndexService;
import java.util.List;
import java.util.Objects;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

/** REST API controller managing vector index lifecycles, insertions, queries, and persistence. */
@RestController
@RequestMapping("/api/v1/indexes")
public class IndexController {

  private final IndexService indexService;

  public IndexController(IndexService indexService) {
    this.indexService = Objects.requireNonNull(indexService, "indexService must not be null");
  }

  /** Creates and registers a new vector index. */
  @PostMapping
  public ResponseEntity<IndexSummaryResponse> createIndex(@RequestBody CreateIndexRequest request) {
    IndexSummaryResponse created = indexService.createIndex(request);
    return ResponseEntity.status(HttpStatus.CREATED).body(created);
  }

  /** Lists all registered vector indices. */
  @GetMapping
  public ResponseEntity<List<IndexSummaryResponse>> listIndexes() {
    List<IndexSummaryResponse> list = indexService.listIndexes();
    return ResponseEntity.ok(list);
  }

  /** Retrieves metadata and status for a specific vector index. */
  @GetMapping("/{name}")
  public ResponseEntity<IndexSummaryResponse> getIndex(@PathVariable("name") String name) {
    IndexSummaryResponse summary = indexService.getIndex(name);
    return ResponseEntity.ok(summary);
  }

  /** Deletes an index and disposes of all underlying allocated resources. */
  @DeleteMapping("/{name}")
  @ResponseStatus(HttpStatus.NO_CONTENT)
  public void deleteIndex(@PathVariable("name") String name) {
    indexService.deleteIndex(name);
  }

  /** Inserts a batch of vectors into a managed index under exclusive write lock. */
  @PostMapping("/{name}/vectors")
  public ResponseEntity<InsertVectorsResponse> insertVectors(
      @PathVariable("name") String name, @RequestBody InsertVectorsRequest request) {
    InsertVectorsResponse response = indexService.insertVectors(name, request);
    return ResponseEntity.ok(response);
  }

  /** Executes a k-NN similarity search over an index under shared read lock. */
  @PostMapping("/{name}/query")
  public ResponseEntity<QueryResponse> query(
      @PathVariable("name") String name, @RequestBody QueryRequest request) {
    QueryResponse response = indexService.query(name, request);
    return ResponseEntity.ok(response);
  }

  /** Persists an in-memory index to disk in NVEC v1 binary format. */
  @PostMapping("/{name}/save")
  public ResponseEntity<SaveIndexResponse> saveIndex(
      @PathVariable("name") String name, @RequestBody(required = false) SaveIndexRequest request) {
    SaveIndexResponse response = indexService.saveIndex(name, request);
    return ResponseEntity.ok(response);
  }

  /** Loads a persisted NVEC v1 binary index file into the registry. */
  @PostMapping("/load")
  public ResponseEntity<IndexSummaryResponse> loadIndex(@RequestBody LoadIndexRequest request) {
    IndexSummaryResponse response = indexService.loadIndex(request);
    return ResponseEntity.status(HttpStatus.CREATED).body(response);
  }
}
