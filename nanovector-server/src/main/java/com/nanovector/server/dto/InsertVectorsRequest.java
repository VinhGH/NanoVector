package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;

/** Request model for inserting a batch of vectors into a managed index. */
@Schema(description = "Batch vector insertion request payload")
public record InsertVectorsRequest(
    @Schema(description = "List of vector items to insert (up to max-batch-size limit)")
        List<VectorItem> vectors) {}
