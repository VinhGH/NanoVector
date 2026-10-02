package com.nanovector.server.dto;

import java.time.Instant;
import java.util.Map;

/** Metadata summary descriptor response for a registered vector index. */
public record IndexSummaryResponse(
    String name,
    String type,
    int dimension,
    String metric,
    int size,
    Instant createdAt,
    Instant lastModifiedAt,
    Map<String, Object> parameters) {}
