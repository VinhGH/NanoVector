package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;

/** Request model for loading a persisted index (.nvec) from the server data directory. */
@Schema(description = "Index loading request specification")
public record LoadIndexRequest(
    @Schema(
            description = "Unique index name to register under in the server registry",
            example = "products_restored")
        String name,
    @Schema(
            description =
                "Source NVEC v1 filename located inside server storage directory (defaults to '{name}.nvec')",
            example = "products.nvec")
        String fileName) {}
