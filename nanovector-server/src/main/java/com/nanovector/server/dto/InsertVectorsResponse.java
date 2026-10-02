package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;

/** Response model returning count of inserted vectors and updated total size. */
@Schema(description = "Batch insertion outcome descriptor")
public record InsertVectorsResponse(
    @Schema(description = "Number of vectors successfully inserted in this request", example = "2")
        int insertedCount,
    @Schema(description = "Total number of active vectors currently in the index", example = "1002")
        int totalSize) {}
