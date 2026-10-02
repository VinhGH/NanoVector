package com.nanovector.cli.command;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.VectorIndex;
import com.nanovector.persistence.format.IndexType;
import com.nanovector.persistence.writer.NvecWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.Callable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Spec;

/**
 * CLI command to initialize and persist a new empty vector index (.nvec file). Supports both FLAT
 * and HNSW topologies with configurable metrics and dimensions.
 */
@Command(
    name = "create",
    mixinStandardHelpOptions = true,
    description = "Create a new vector index (.nvec) file.")
public class CreateCommand implements Callable<Integer> {

  @Spec private CommandSpec spec;

  @Parameters(index = "0", description = "Target .nvec file path")
  private Path targetPath;

  @Option(
      names = {"-t", "--type"},
      defaultValue = "HNSW",
      description = "Index type: FLAT or HNSW (default: HNSW)")
  private String typeStr;

  @Option(
      names = {"-d", "--dimension"},
      required = true,
      description = "Vector dimension (> 0)")
  private int dimension;

  @Option(
      names = {"-m", "--metric"},
      defaultValue = "EUCLIDEAN",
      description = "Distance metric: EUCLIDEAN, COSINE, DOT_PRODUCT (default: EUCLIDEAN)")
  private String metricStr;

  @Option(
      names = {"--hnsw-m"},
      defaultValue = "16",
      description = "HNSW M: maximum outgoing edges per node (default: 16)")
  private int m;

  @Option(
      names = {"--hnsw-m0"},
      description = "HNSW M0: maximum outgoing edges at layer 0 (default: 2 * M)")
  private Integer m0;

  @Option(
      names = {"--hnsw-ef-construction"},
      defaultValue = "200",
      description = "HNSW efConstruction: construction dynamic candidate list size (default: 200)")
  private int efConstruction;

  @Option(
      names = {"--hnsw-ef-search"},
      defaultValue = "50",
      description = "HNSW efSearch: default search dynamic candidate list size (default: 50)")
  private int efSearch;

  @Option(
      names = {"-f", "--force"},
      defaultValue = "false",
      description = "Overwrite existing file if present")
  private boolean force;

  public CreateCommand() {}

  public CreateCommand(Path targetPath, String typeStr, int dimension, String metricStr) {
    this.targetPath = targetPath;
    this.typeStr = typeStr;
    this.dimension = dimension;
    this.metricStr = metricStr;
    this.m = 16;
    this.efConstruction = 200;
    this.efSearch = 50;
  }

  @Override
  public Integer call() {
    PrintWriter out =
        spec != null ? spec.commandLine().getOut() : new PrintWriter(System.out, true);
    PrintWriter err =
        spec != null ? spec.commandLine().getErr() : new PrintWriter(System.err, true);

    if (targetPath == null) {
      err.println("Error: Missing target file path argument.");
      return 1;
    }

    if (dimension <= 0) {
      err.println("Error: Dimension must be positive, got: " + dimension);
      return 1;
    }

    DistanceMetric metric;
    try {
      metric = DistanceMetric.valueOf(metricStr.toUpperCase(Locale.ROOT));
    } catch (IllegalArgumentException e) {
      err.println(
          "Error: Unsupported distance metric '"
              + metricStr
              + "'. Supported: EUCLIDEAN, COSINE, DOT_PRODUCT");
      return 1;
    }

    IndexType indexType;
    try {
      indexType = IndexType.valueOf(typeStr.toUpperCase(Locale.ROOT));
    } catch (IllegalArgumentException e) {
      err.println("Error: Unsupported index type '" + typeStr + "'. Supported: FLAT, HNSW");
      return 1;
    }

    if (Files.exists(targetPath) && !force) {
      err.println(
          "Error: Target file already exists: " + targetPath + ". Use --force to overwrite.");
      return 1;
    }

    VectorIndex index;
    try {
      if (indexType == IndexType.FLAT) {
        index = new FlatIndex(dimension, metric);
      } else {
        int actualM0 = (m0 != null) ? m0 : (2 * m);
        if (m < 2) {
          err.println("Error: HNSW M must be at least 2, got: " + m);
          return 1;
        }
        if (actualM0 < m) {
          err.println("Error: HNSW M0 must be >= M (" + m + "), got: " + actualM0);
          return 1;
        }
        if (efConstruction <= 0) {
          err.println("Error: HNSW efConstruction must be positive, got: " + efConstruction);
          return 1;
        }
        if (efSearch <= 0) {
          err.println("Error: HNSW efSearch must be positive, got: " + efSearch);
          return 1;
        }
        HnswConfig config =
            new HnswConfig(m, actualM0, efConstruction, efSearch, 1.0 / Math.log(m), null);
        index = new HnswIndex(dimension, metric, config);
      }
    } catch (Exception e) {
      err.println("Error configuring index: " + e.getMessage());
      return 1;
    }

    try {
      NvecWriter.write(index, targetPath);
      out.printf(
          Locale.ROOT,
          "Successfully created %s index [%s, Dimension: %d] at %s%n",
          indexType,
          metric,
          dimension,
          targetPath.toAbsolutePath().normalize());
      return 0;
    } catch (IOException e) {
      err.println("Error writing index file: " + e.getMessage());
      return 1;
    }
  }
}
