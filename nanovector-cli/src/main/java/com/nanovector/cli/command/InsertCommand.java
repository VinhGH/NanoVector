package com.nanovector.cli.command;

import com.nanovector.core.index.VectorIndex;
import com.nanovector.core.storage.VectorDataView;
import com.nanovector.persistence.reader.NvecReader;
import com.nanovector.persistence.writer.NvecWriter;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.Callable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Spec;

/**
 * CLI command to ingest vector records from a CSV file into an existing .nvec index. Guarantees
 * all-or-nothing transactional semantics and atomic persistence.
 */
@Command(
    name = "insert",
    mixinStandardHelpOptions = true,
    description = "Insert vectors from a CSV file into an existing .nvec index.")
public class InsertCommand implements Callable<Integer> {

  @Spec private CommandSpec spec;

  @Parameters(index = "0", description = "Target .nvec index file path")
  private Path targetPath;

  @Option(
      names = {"-f", "--file"},
      required = true,
      description = "Path to the CSV file containing vectors")
  private Path csvPath;

  record CsvEntry(long id, float[] vector, int lineNumber) {}

  public InsertCommand() {}

  public InsertCommand(Path targetPath, Path csvPath) {
    this.targetPath = targetPath;
    this.csvPath = csvPath;
  }

  @Override
  public Integer call() {
    PrintWriter out =
        spec != null ? spec.commandLine().getOut() : new PrintWriter(System.out, true);
    PrintWriter err =
        spec != null ? spec.commandLine().getErr() : new PrintWriter(System.err, true);

    if (targetPath == null) {
      err.println("Error: Missing target index file path.");
      return 1;
    }

    if (csvPath == null) {
      err.println("Error: Missing CSV file path option (-f/--file).");
      return 1;
    }

    if (!Files.exists(targetPath)) {
      err.println("Error: Index file not found: " + targetPath);
      return 1;
    }

    if (!Files.exists(csvPath)) {
      err.println("Error: CSV file not found: " + csvPath);
      return 1;
    }

    VectorIndex index;
    try {
      index = NvecReader.read(targetPath);
    } catch (Exception e) {
      err.println("Error loading index file: " + e.getMessage());
      return 1;
    }

    int dimension = index.dimension();
    VectorDataView view = index.vectorData();
    Set<Long> existingIds = new HashSet<>(view.size());
    for (int i = 0; i < view.size(); i++) {
      existingIds.add(view.getExternalId(i));
    }

    // Step 1: Parse and validate entire CSV in memory upfront (Atomicity Gate)
    List<CsvEntry> entries = new ArrayList<>();
    Set<Long> batchIds = new HashSet<>();

    try (BufferedReader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
      String line;
      int lineNumber = 0;
      boolean firstRow = true;

      while ((line = reader.readLine()) != null) {
        lineNumber++;
        String trimmed = line.trim();
        if (trimmed.isEmpty() || trimmed.startsWith("#")) {
          continue;
        }

        // Header detection on first non-empty line
        if (firstRow) {
          firstRow = false;
          if (isHeaderLine(trimmed)) {
            continue;
          }
        }

        CsvEntry entry = parseLine(trimmed, lineNumber, dimension, existingIds, batchIds);
        entries.add(entry);
      }
    } catch (CsvValidationException e) {
      err.println("Error: " + e.getMessage());
      return 1;
    } catch (IOException e) {
      err.println("Error reading CSV file: " + e.getMessage());
      return 1;
    }

    if (entries.isEmpty()) {
      out.println("Warning: CSV file contained no vector data rows. Index unchanged.");
      return 0;
    }

    // Step 2: Ingest validated batch into index
    try {
      for (CsvEntry entry : entries) {
        index.insert(entry.id(), entry.vector());
      }
    } catch (Exception e) {
      err.println("Error inserting vectors: " + e.getMessage());
      return 1;
    }

    // Step 3: Persist updated index atomically (via temp file & atomic move)
    try {
      NvecWriter.write(index, targetPath);
      out.printf(
          Locale.ROOT,
          "Successfully inserted %,d vectors into %s (Total vector count: %,d)%n",
          entries.size(),
          targetPath.toAbsolutePath().normalize(),
          index.size());
      return 0;
    } catch (IOException e) {
      err.println("Error persisting updated index to file: " + e.getMessage());
      return 1;
    }
  }

  private boolean isHeaderLine(String line) {
    int comma = line.indexOf(',');
    if (comma <= 0) {
      return false;
    }
    String firstCol = line.substring(0, comma).trim();
    try {
      Long.parseLong(firstCol);
      return false; // numeric ID -> data row
    } catch (NumberFormatException e) {
      return true; // non-numeric header (e.g. "id", "ID")
    }
  }

  private CsvEntry parseLine(
      String line, int lineNumber, int expectedDim, Set<Long> existingIds, Set<Long> batchIds)
      throws CsvValidationException {
    int comma = line.indexOf(',');
    if (comma <= 0) {
      throw new CsvValidationException(
          "Line "
              + lineNumber
              + ": Invalid CSV format, missing comma separating ID and vector: '"
              + line
              + "'",
          lineNumber);
    }

    String idPart = line.substring(0, comma).trim();
    String vecPart = line.substring(comma + 1).trim();

    long id;
    try {
      id = Long.parseLong(idPart);
    } catch (NumberFormatException e) {
      throw new CsvValidationException(
          "Line " + lineNumber + ": Invalid ID '" + idPart + "' - must be an integer (long)",
          lineNumber);
    }

    if (!batchIds.add(id)) {
      throw new CsvValidationException(
          "Line " + lineNumber + ": Duplicate ID " + id + " found within CSV batch", lineNumber);
    }

    if (existingIds.contains(id)) {
      throw new CsvValidationException(
          "Line " + lineNumber + ": ID " + id + " already exists in index", lineNumber);
    }

    // Strip optional enclosing quotes and brackets
    if (vecPart.startsWith("\"") && vecPart.endsWith("\"") && vecPart.length() >= 2) {
      vecPart = vecPart.substring(1, vecPart.length() - 1).trim();
    }
    if (vecPart.startsWith("[") && vecPart.endsWith("]") && vecPart.length() >= 2) {
      vecPart = vecPart.substring(1, vecPart.length() - 1).trim();
    }

    if (vecPart.isEmpty()) {
      throw new CsvValidationException("Line " + lineNumber + ": Vector data is empty", lineNumber);
    }

    String[] tokens = vecPart.split("[,\\s]+");
    if (tokens.length != expectedDim) {
      throw new CsvValidationException(
          "Line "
              + lineNumber
              + ": Dimension mismatch (expected "
              + expectedDim
              + ", got "
              + tokens.length
              + ")",
          lineNumber);
    }

    float[] vector = new float[expectedDim];
    for (int d = 0; d < expectedDim; d++) {
      float val;
      try {
        val = Float.parseFloat(tokens[d]);
      } catch (NumberFormatException e) {
        throw new CsvValidationException(
            "Line "
                + lineNumber
                + ": Invalid float value at dimension "
                + d
                + ": '"
                + tokens[d]
                + "'",
            lineNumber);
      }
      if (!Float.isFinite(val)) {
        throw new CsvValidationException(
            "Line "
                + lineNumber
                + ": Non-finite float value (NaN or Infinity) at dimension "
                + d
                + ": "
                + val,
            lineNumber);
      }
      vector[d] = val;
    }

    return new CsvEntry(id, vector, lineNumber);
  }
}
