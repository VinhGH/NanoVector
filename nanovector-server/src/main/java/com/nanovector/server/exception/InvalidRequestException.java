package com.nanovector.server.exception;

/** Thrown when request arguments or payloads fail domain validation invariants. */
public class InvalidRequestException extends RuntimeException {

  public InvalidRequestException(String message) {
    super(message);
  }
}
