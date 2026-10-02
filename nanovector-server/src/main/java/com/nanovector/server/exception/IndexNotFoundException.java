package com.nanovector.server.exception;

/** Thrown when an operation targets an index that does not exist in the registry. */
public class IndexNotFoundException extends RuntimeException {

  public IndexNotFoundException(String message) {
    super(message);
  }
}
