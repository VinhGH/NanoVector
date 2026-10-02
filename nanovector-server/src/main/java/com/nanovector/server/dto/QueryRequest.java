package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;

/** Request model for k-NN similarity search over a vector index. */
@Schema(description = "k-NN similarity query request")
public record QueryRequest(
    @Schema(
            description = "Query vector float array matching index dimension",
            example = "[0.12, 0.25, 0.38]")
        float[] vector,
    @Schema(
            description = "Number of nearest neighbors to retrieve (default: 10, positive)",
            example = "10")
        Integer k,
    @Schema(
            description =
                "Exploration candidate list width for graph-based indexes (efSearch override, positive)",
            example = "50")
        Integer efSearch) {}
