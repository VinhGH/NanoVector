package com.nanovector.cli.command;

/** Exception thrown when CSV data format or invariant validation fails during batch parsing. */
public class CsvValidationException extends Exception {

  private final int lineNumber;

  public CsvValidationException(String message, int lineNumber) {
    super(message);
    this.lineNumber = lineNumber;
  }

  public int getLineNumber() {
    return lineNumber;
  }
}
