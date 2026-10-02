package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;

/** Individual vector item container inside an insert request. */
@Schema(description = "Individual vector payload item")
public record VectorItem(
    @Schema(
            description = "Unique external vector identifier (non-negative integer)",
            example = "1001")
        Long id,
    @Schema(
            description = "Primitive float vector components matching index dimension",
            example = "[0.12, 0.25, 0.38]")
        float[] values) {}
