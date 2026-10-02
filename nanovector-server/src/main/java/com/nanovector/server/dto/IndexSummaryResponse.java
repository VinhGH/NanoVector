package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.time.Instant;
import java.util.Map;

/** Metadata summary descriptor response for a registered vector index. */
@Schema(description = "Index metadata and operational summary")
public record IndexSummaryResponse(
    @Schema(description = "Index name", example = "products") String name,
    @Schema(description = "Index topology type", example = "HNSW") String type,
    @Schema(description = "Vector dimensionality", example = "128") int dimension,
    @Schema(description = "Distance metric", example = "EUCLIDEAN") String metric,
    @Schema(description = "Current number of vectors in index", example = "1000") int size,
    @Schema(description = "Creation timestamp (ISO-8601)", example = "2026-10-02T10:00:00Z")
        Instant createdAt,
    @Schema(
            description = "Last modification timestamp (ISO-8601)",
            example = "2026-10-02T10:05:00Z")
        Instant lastModifiedAt,
    @Schema(
            description = "Hyperparameters and topology settings",
            example = "{\"m\": 16, \"efConstruction\": 200, \"efSearch\": 50}")
        Map<String, Object> parameters) {}
