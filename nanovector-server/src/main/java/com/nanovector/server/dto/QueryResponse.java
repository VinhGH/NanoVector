package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;

/** Response model returning k-NN similarity search results and execution metrics. */
@Schema(description = "k-NN query response payload")
public record QueryResponse(
    @Schema(description = "Queried index name", example = "products") String indexName,
    @Schema(description = "Requested k limit", example = "10") int k,
    @Schema(description = "Ranked list of nearest neighbors ordered by distance ascending")
        List<SearchResultItem> results,
    @Schema(description = "Execution duration in microseconds", example = "142") long tookMicros) {}
