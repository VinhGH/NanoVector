package com.nanovector.cli.command;

import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.VectorIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.persistence.reader.NvecReader;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Spec;

/**
 * CLI command to execute Top-K nearest neighbor search against an existing .nvec index. Supports
 * configurable k and efSearch parameter override for HNSW graphs.
 */
@Command(
    name = "query",
    mixinStandardHelpOptions = true,
    description = "Query k-NN nearest vectors against an existing .nvec index.")
public class QueryCommand implements Callable<Integer> {

  @Spec private CommandSpec spec;

  @Parameters(index = "0", description = "Target .nvec index file path")
  private Path targetPath;

  @Option(
      names = {"-k", "--top-k"},
      defaultValue = "10",
      description = "Number of nearest neighbors to return (default: 10)")
  private int k;

  @Option(
      names = {"-v", "--vector"},
      required = true,
      description = "Query vector as comma- or space-separated floats")
  private String vectorStr;

  @Option(
      names = {"--ef-search"},
      description = "HNSW efSearch parameter override (HNSW index only)")
  private Integer efSearch;

  public QueryCommand() {}

  public QueryCommand(Path targetPath, int k, String vectorStr) {
    this.targetPath = targetPath;
    this.k = k;
    this.vectorStr = vectorStr;
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

    if (!Files.exists(targetPath)) {
      err.println("Error: Index file not found: " + targetPath);
      return 1;
    }

    if (k <= 0) {
      err.println("Error: k must be positive, got: " + k);
      return 1;
    }

    if (efSearch != null && efSearch <= 0) {
      err.println("Error: efSearch must be positive, got: " + efSearch);
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

    float[] queryVector;
    try {
      queryVector = parseVector(vectorStr, dimension);
    } catch (IllegalArgumentException e) {
      err.println("Error: " + e.getMessage());
      return 1;
    }

    if (index.size() == 0) {
      out.println("Index is empty (0 vectors). No results found.");
      return 0;
    }

    long startNanos = System.nanoTime();
    List<SearchResult> results;
    if (index instanceof HnswIndex hnsw && efSearch != null) {
      results = hnsw.searchKnn(queryVector, k, efSearch);
    } else {
      results = index.searchKnn(queryVector, k);
    }
    long elapsedNanos = System.nanoTime() - startNanos;
    double elapsedMs = elapsedNanos / 1_000_000.0;

    printResults(out, results, elapsedMs);
    return 0;
  }

  private float[] parseVector(String raw, int expectedDim) {
    if (raw == null) {
      throw new IllegalArgumentException("Query vector must not be null");
    }
    String cleaned = raw.trim();
    if ((cleaned.startsWith("\"") && !cleaned.endsWith("\""))
        || (!cleaned.startsWith("\"") && cleaned.endsWith("\""))) {
      throw new IllegalArgumentException("Unclosed quote in query vector: '" + cleaned + "'");
    }
    if ((cleaned.startsWith("[") && !cleaned.endsWith("]"))
        || (!cleaned.startsWith("[") && cleaned.endsWith("]"))) {
      throw new IllegalArgumentException("Unclosed bracket in query vector: '" + cleaned + "'");
    }
    if (cleaned.startsWith("\"") && cleaned.endsWith("\"") && cleaned.length() >= 2) {
      cleaned = cleaned.substring(1, cleaned.length() - 1).trim();
    }
    if (cleaned.startsWith("[") && cleaned.endsWith("]") && cleaned.length() >= 2) {
      cleaned = cleaned.substring(1, cleaned.length() - 1).trim();
    }
    if (cleaned.isEmpty()) {
      throw new IllegalArgumentException("Query vector string is empty");
    }

    String[] tokens = cleaned.split("[,\\s]+");
    if (tokens.length != expectedDim) {
      throw new IllegalArgumentException(
          "Query vector dimension mismatch: expected " + expectedDim + ", got " + tokens.length);
    }

    float[] vec = new float[expectedDim];
    for (int i = 0; i < expectedDim; i++) {
      try {
        vec[i] = Float.parseFloat(tokens[i]);
      } catch (NumberFormatException e) {
        throw new IllegalArgumentException("Invalid float in query vector: '" + tokens[i] + "'");
      }
      if (!Float.isFinite(vec[i])) {
        throw new IllegalArgumentException(
            "Query vector contains non-finite float (NaN or Infinity): " + vec[i]);
      }
    }
    return vec;
  }

  private void printResults(PrintWriter out, List<SearchResult> results, double elapsedMs) {
    out.println("================================================================================");
    out.printf(
        Locale.ROOT, "Query Results (Top-%d, Latency: %.3f ms)%n", results.size(), elapsedMs);
    out.println("================================================================================");
    out.printf(Locale.ROOT, "%-8s %-24s %-20s%n", "Rank", "External ID", "Distance");
    out.println("--------------------------------------------------------------------------------");
    for (int i = 0; i < results.size(); i++) {
      SearchResult r = results.get(i);
      out.printf(Locale.ROOT, "%-8d %-24d %-20.6f%n", (i + 1), r.id(), r.distance());
    }
    out.println("================================================================================");
    out.flush();
  }
}
