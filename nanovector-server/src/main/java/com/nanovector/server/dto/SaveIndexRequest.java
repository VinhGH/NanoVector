package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;

/** Request model for persisting an in-memory index to disk in NVEC v1 format. */
@Schema(description = "Index persistence request")
public record SaveIndexRequest(
    @Schema(
            description =
                "Optional safe destination filename in data/indexes (defaults to '{name}.nvec')",
            example = "products.nvec")
        String fileName) {}
