package com.nanovector.server.dto;

import java.time.Instant;

/** Standard error response model for REST API endpoints. */
public record ErrorResponse(
    String timestamp, int status, String error, String message, String path) {

  public static ErrorResponse of(int status, String error, String message, String path) {
    return new ErrorResponse(Instant.now().toString(), status, error, message, path);
  }
}
