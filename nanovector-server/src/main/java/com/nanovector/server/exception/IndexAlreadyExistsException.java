package com.nanovector.server.exception;

/** Thrown when attempting to register or create an index whose name already exists. */
public class IndexAlreadyExistsException extends RuntimeException {

  public IndexAlreadyExistsException(String message) {
    super(message);
  }
}
