package com.nanovector.server.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.time.Instant;

/** Standard error response model for REST API endpoints. */
@Schema(description = "Standardized REST API error payload")
public record ErrorResponse(
    @Schema(description = "Timestamp of the error (ISO-8601)", example = "2026-10-02T10:15:30Z")
        String timestamp,
    @Schema(description = "HTTP status code", example = "400") int status,
    @Schema(description = "HTTP error reason phrase", example = "Bad Request") String error,
    @Schema(
            description = "Detailed error message or validation invariant violation",
            example = "Vector dimension mismatch: expected 128, got 64")
        String message,
    @Schema(description = "Target API request URI", example = "/api/v1/indexes/products/vectors")
        String path) {

  public static ErrorResponse of(int status, String error, String message, String path) {
    return new ErrorResponse(Instant.now().toString(), status, error, message, path);
  }
}
