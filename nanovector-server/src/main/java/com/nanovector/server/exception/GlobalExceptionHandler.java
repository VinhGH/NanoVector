package com.nanovector.server.exception;

import com.nanovector.persistence.exception.CorruptIndexException;
import com.nanovector.server.dto.ErrorResponse;
import jakarta.servlet.http.HttpServletRequest;
import java.io.FileNotFoundException;
import java.util.NoSuchElementException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * Central exception translator for REST API controllers, producing standardized error responses.
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

  private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

  @ExceptionHandler({
    InvalidRequestException.class,
    IllegalArgumentException.class,
    HttpMessageNotReadableException.class,
    CorruptIndexException.class,
    UnsupportedOperationException.class
  })
  public ResponseEntity<ErrorResponse> handleBadRequest(Exception ex, HttpServletRequest request) {
    log.debug("Bad request on {}: {}", request.getRequestURI(), ex.getMessage());
    return buildResponse(HttpStatus.BAD_REQUEST, ex.getMessage(), request);
  }

  @ExceptionHandler({
    IndexNotFoundException.class,
    NoSuchElementException.class,
    FileNotFoundException.class
  })
  public ResponseEntity<ErrorResponse> handleNotFound(Exception ex, HttpServletRequest request) {
    log.debug("Resource not found on {}: {}", request.getRequestURI(), ex.getMessage());
    return buildResponse(HttpStatus.NOT_FOUND, ex.getMessage(), request);
  }

  @ExceptionHandler(IndexAlreadyExistsException.class)
  public ResponseEntity<ErrorResponse> handleConflict(
      IndexAlreadyExistsException ex, HttpServletRequest request) {
    log.debug("Resource conflict on {}: {}", request.getRequestURI(), ex.getMessage());
    return buildResponse(HttpStatus.CONFLICT, ex.getMessage(), request);
  }

  @ExceptionHandler(IllegalStateException.class)
  public ResponseEntity<ErrorResponse> handleIllegalState(
      IllegalStateException ex, HttpServletRequest request) {
    log.debug("Illegal state on {}: {}", request.getRequestURI(), ex.getMessage());
    return buildResponse(HttpStatus.CONFLICT, ex.getMessage(), request);
  }

  @ExceptionHandler(PayloadTooLargeException.class)
  public ResponseEntity<ErrorResponse> handlePayloadTooLarge(
      PayloadTooLargeException ex, HttpServletRequest request) {
    log.debug("Payload too large on {}: {}", request.getRequestURI(), ex.getMessage());
    return buildResponse(HttpStatus.PAYLOAD_TOO_LARGE, ex.getMessage(), request);
  }

  @ExceptionHandler(Exception.class)
  public ResponseEntity<ErrorResponse> handleGeneralException(
      Exception ex, HttpServletRequest request) {
    log.error(
        "Unhandled exception processing {}: {}", request.getRequestURI(), ex.getMessage(), ex);
    return buildResponse(
        HttpStatus.INTERNAL_SERVER_ERROR,
        "An unexpected error occurred: " + ex.getMessage(),
        request);
  }

  private ResponseEntity<ErrorResponse> buildResponse(
      HttpStatus status, String message, HttpServletRequest request) {
    String safeMsg = message != null ? message : status.getReasonPhrase();
    ErrorResponse error =
        ErrorResponse.of(
            status.value(), status.getReasonPhrase(), safeMsg, request.getRequestURI());
    return ResponseEntity.status(status).body(error);
  }
}
