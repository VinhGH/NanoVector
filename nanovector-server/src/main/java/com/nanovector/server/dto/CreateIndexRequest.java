package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.Map;

/** Request model for creating a new vector index. */
@Schema(description = "Index creation specification")
public record CreateIndexRequest(
    @Schema(
            description = "Unique alphanumeric index name (1-64 characters, letters, digits, _, -)",
            example = "products")
        String name,
    @Schema(
            description = "Index topology type",
            example = "HNSW",
            allowableValues = {"FLAT", "HNSW", "HNSW_SQ8", "HNSW_SQ8_OFFHEAP"})
        String type,
    @Schema(description = "Vector dimensionality (positive integer <= 4096)", example = "128")
        Integer dimension,
    @Schema(
            description = "Distance metric",
            example = "EUCLIDEAN",
            allowableValues = {"EUCLIDEAN", "COSINE", "DOT_PRODUCT"})
        String metric,
    @Schema(
            description =
                "Engine hyperparameters (e.g., m, m0, efConstruction, efSearch for HNSW topologies)",
            example = "{\"m\": 16, \"efConstruction\": 200, \"efSearch\": 50}")
        Map<String, Object> parameters) {}
