package com.nanovector.server.exception;

/** Thrown when an incoming batch payload exceeds maximum allowed capacity limits. */
public class PayloadTooLargeException extends RuntimeException {

  public PayloadTooLargeException(String message) {
    super(message);
  }
}
