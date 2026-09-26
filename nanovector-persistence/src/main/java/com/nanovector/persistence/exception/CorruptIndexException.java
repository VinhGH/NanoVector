package com.nanovector.persistence.exception;

import java.io.IOException;

/**
 * Thrown when an NVEC index file is malformed, truncated, or fails checksum/invariant validation.
 */
public class CorruptIndexException extends IOException {

  public CorruptIndexException(String message) {
    super(message);
  }

  public CorruptIndexException(String message, Throwable cause) {
    super(message, cause);
  }
}
