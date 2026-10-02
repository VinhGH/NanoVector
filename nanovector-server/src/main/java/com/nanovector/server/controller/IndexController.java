package com.nanovector.server.controller;

import com.nanovector.server.dto.CreateIndexRequest;
import com.nanovector.server.dto.ErrorResponse;
import com.nanovector.server.dto.IndexSummaryResponse;
import com.nanovector.server.dto.InsertVectorsRequest;
import com.nanovector.server.dto.InsertVectorsResponse;
import com.nanovector.server.dto.LoadIndexRequest;
import com.nanovector.server.dto.QueryRequest;
import com.nanovector.server.dto.QueryResponse;
import com.nanovector.server.dto.SaveIndexRequest;
import com.nanovector.server.dto.SaveIndexResponse;
import com.nanovector.server.service.IndexService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
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
@Tag(
    name = "Vector Indexes",
    description =
        "Lifecycle management, batch ingestion, k-NN similarity search, and NVEC persistence operations")
public class IndexController {

  private final IndexService indexService;

  public IndexController(IndexService indexService) {
    this.indexService = Objects.requireNonNull(indexService, "indexService must not be null");
  }

  @Operation(
      summary = "Create vector index",
      description =
          "Initializes and registers a new vector index with specified topology, dimensionality, and distance metric.")
  @ApiResponses({
    @ApiResponse(
        responseCode = "201",
        description = "Index created successfully",
        content = @Content(schema = @Schema(implementation = IndexSummaryResponse.class))),
    @ApiResponse(
        responseCode = "400",
        description = "Invalid configuration, dimension mismatch, or unsupported metric",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
    @ApiResponse(
        responseCode = "409",
        description = "Index with specified name already exists",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
  })
  @PostMapping
  public ResponseEntity<IndexSummaryResponse> createIndex(@RequestBody CreateIndexRequest request) {
    IndexSummaryResponse created = indexService.createIndex(request);
    return ResponseEntity.status(HttpStatus.CREATED).body(created);
  }

  @Operation(
      summary = "List all vector indexes",
      description = "Retrieves metadata snapshots of all registered active vector indexes.")
  @ApiResponse(
      responseCode = "200",
      description = "List of active indexes",
      content =
          @Content(
              array = @ArraySchema(schema = @Schema(implementation = IndexSummaryResponse.class))))
  @GetMapping
  public ResponseEntity<List<IndexSummaryResponse>> listIndexes() {
    List<IndexSummaryResponse> list = indexService.listIndexes();
    return ResponseEntity.ok(list);
  }

  @Operation(
      summary = "Get index metadata",
      description = "Retrieves detailed metadata, capacity, and operational status for an index.")
  @ApiResponses({
    @ApiResponse(
        responseCode = "200",
        description = "Index metadata descriptor",
        content = @Content(schema = @Schema(implementation = IndexSummaryResponse.class))),
    @ApiResponse(
        responseCode = "404",
        description = "Index not found",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
  })
  @GetMapping("/{name}")
  public ResponseEntity<IndexSummaryResponse> getIndex(
      @Parameter(description = "Index name identifier", example = "products") @PathVariable("name")
          String name) {
    IndexSummaryResponse summary = indexService.getIndex(name);
    return ResponseEntity.ok(summary);
  }

  @Operation(
      summary = "Delete vector index",
      description =
          "Gracefully deletes the index, draining in-flight queries and releasing off-heap native memory segments.")
  @ApiResponses({
    @ApiResponse(responseCode = "204", description = "Index successfully deleted and closed"),
    @ApiResponse(
        responseCode = "404",
        description = "Index not found",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
  })
  @DeleteMapping("/{name}")
  @ResponseStatus(HttpStatus.NO_CONTENT)
  public void deleteIndex(
      @Parameter(description = "Index name identifier", example = "products") @PathVariable("name")
          String name) {
    indexService.deleteIndex(name);
  }

  @Operation(
      summary = "Insert batch vectors",
      description =
          "Inserts a batch of vectors under exclusive write lock protection with strict pre-validation.")
  @ApiResponses({
    @ApiResponse(
        responseCode = "200",
        description = "Vectors inserted successfully",
        content = @Content(schema = @Schema(implementation = InsertVectorsResponse.class))),
    @ApiResponse(
        responseCode = "400",
        description = "Validation failure, dimension mismatch, or duplicate external ID",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
    @ApiResponse(
        responseCode = "404",
        description = "Index not found",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
    @ApiResponse(
        responseCode = "413",
        description = "Batch vector count exceeds max-batch-size limit",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
  })
  @PostMapping("/{name}/vectors")
  public ResponseEntity<InsertVectorsResponse> insertVectors(
      @Parameter(description = "Index name identifier", example = "products") @PathVariable("name")
          String name,
      @RequestBody InsertVectorsRequest request) {
    InsertVectorsResponse response = indexService.insertVectors(name, request);
    return ResponseEntity.ok(response);
  }

  @Operation(
      summary = "Execute k-NN query",
      description =
          "Performs approximate or exact nearest neighbor search holding a shared read lock with isolated visited context.")
  @ApiResponses({
    @ApiResponse(
        responseCode = "200",
        description = "Query executed successfully",
        content = @Content(schema = @Schema(implementation = QueryResponse.class))),
    @ApiResponse(
        responseCode = "400",
        description = "Query vector dimension mismatch, NaN/Inf values, or invalid k/efSearch",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
    @ApiResponse(
        responseCode = "404",
        description = "Index not found",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
  })
  @PostMapping("/{name}/query")
  public ResponseEntity<QueryResponse> query(
      @Parameter(description = "Index name identifier", example = "products") @PathVariable("name")
          String name,
      @RequestBody QueryRequest request) {
    QueryResponse response = indexService.query(name, request);
    return ResponseEntity.ok(response);
  }

  @Operation(
      summary = "Persist index to NVEC v1",
      description =
          "Serializes the index to an NVEC v1 binary file in the server data directory, holding the read lock throughout serialization.")
  @ApiResponses({
    @ApiResponse(
        responseCode = "200",
        description = "Index successfully serialized",
        content = @Content(schema = @Schema(implementation = SaveIndexResponse.class))),
    @ApiResponse(
        responseCode = "400",
        description = "Index topology not supported for persistence or invalid filename",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
    @ApiResponse(
        responseCode = "404",
        description = "Index not found",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
  })
  @PostMapping("/{name}/save")
  public ResponseEntity<SaveIndexResponse> saveIndex(
      @Parameter(description = "Index name identifier", example = "products") @PathVariable("name")
          String name,
      @RequestBody(required = false) SaveIndexRequest request) {
    SaveIndexResponse response = indexService.saveIndex(name, request);
    return ResponseEntity.ok(response);
  }

  @Operation(
      summary = "Load index from NVEC v1",
      description =
          "Restores an index from an NVEC v1 binary file located inside the server storage directory into the registry.")
  @ApiResponses({
    @ApiResponse(
        responseCode = "201",
        description = "Index loaded and registered successfully",
        content = @Content(schema = @Schema(implementation = IndexSummaryResponse.class))),
    @ApiResponse(
        responseCode = "400",
        description = "Corrupted file, checksum mismatch, or invalid parameters",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
    @ApiResponse(
        responseCode = "404",
        description = "Target file not found in storage directory",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
    @ApiResponse(
        responseCode = "409",
        description = "Index with specified registration name already exists",
        content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
  })
  @PostMapping("/load")
  public ResponseEntity<IndexSummaryResponse> loadIndex(@RequestBody LoadIndexRequest request) {
    IndexSummaryResponse response = indexService.loadIndex(request);
    return ResponseEntity.status(HttpStatus.CREATED).body(response);
  }
}
