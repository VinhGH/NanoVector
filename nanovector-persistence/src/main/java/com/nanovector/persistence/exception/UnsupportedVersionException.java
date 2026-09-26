package com.nanovector.persistence.exception;

import java.io.IOException;

/**
 * Thrown when an NVEC file specifies a format version that is unsupported by the current parser.
 */
public class UnsupportedVersionException extends IOException {

  public UnsupportedVersionException(String message) {
    super(message);
  }

  public UnsupportedVersionException(String message, Throwable cause) {
    super(message, cause);
  }
}
