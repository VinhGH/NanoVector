package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;

/** Response model returning persistence outcome and target file details. */
@Schema(description = "Index persistence outcome descriptor")
public record SaveIndexResponse(
    @Schema(description = "Persisted index name", example = "products") String indexName,
    @Schema(description = "Generated NVEC v1 binary filename", example = "products.nvec")
        String fileName,
    @Schema(description = "Number of vectors serialized", example = "1000") int size,
    @Schema(description = "Total size of the generated binary file in bytes", example = "524832")
        long fileSizeBytes) {}
