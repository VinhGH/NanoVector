package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;

/** Individual k-NN search result item containing external vector ID and distance score. */
@Schema(description = "Retrieved nearest neighbor item")
public record SearchResultItem(
    @Schema(description = "External vector ID of the matching item", example = "1001") long id,
    @Schema(
            description = "Calculated distance metric score (smaller is closer)",
            example = "0.0425")
        float score) {}
