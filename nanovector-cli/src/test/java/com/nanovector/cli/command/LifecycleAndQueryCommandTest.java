package com.nanovector.cli.command;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.cli.NanoVectorCli;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.core.index.VectorIndex;
import com.nanovector.persistence.reader.NvecInspectionResult;
import com.nanovector.persistence.reader.NvecReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import picocli.CommandLine;

class LifecycleAndQueryCommandTest {

  @TempDir Path tempDir;

  private ByteArrayOutputStream outStream;
  private ByteArrayOutputStream errStream;
  private CommandLine commandLine;

  @BeforeEach
  void setUp() {
    outStream = new ByteArrayOutputStream();
    errStream = new ByteArrayOutputStream();
    commandLine = NanoVectorCli.createCommandLine();
    commandLine.setOut(new PrintWriter(outStream, true, StandardCharsets.UTF_8));
    commandLine.setErr(new PrintWriter(errStream, true, StandardCharsets.UTF_8));
  }

  private String stdout() {
    return outStream.toString(StandardCharsets.UTF_8);
  }

  private String stderr() {
    return errStream.toString(StandardCharsets.UTF_8);
  }

  // ═════════════════════════════════════════════════════════════════════
  // 1. CREATE COMMAND TESTS
  // ═════════════════════════════════════════════════════════════════════

  @Test
  @DisplayName("Create FLAT index succeeds and initializes valid empty .nvec file")
  void testCreateFlatIndex() throws IOException {
    Path indexFile = tempDir.resolve("test_flat.nvec");

    int exitCode =
        commandLine.execute(
            "create",
            indexFile.toString(),
            "--type",
            "FLAT",
            "--dimension",
            "4",
            "--metric",
            "EUCLIDEAN");

    assertThat(exitCode).isEqualTo(0);
    assertThat(stderr()).isEmpty();
    assertThat(stdout()).contains("Successfully created FLAT index");

    // Verify file on disk using NvecReader
    VectorIndex index = NvecReader.read(indexFile);
    assertThat(index).isInstanceOf(FlatIndex.class);
    assertThat(index.dimension()).isEqualTo(4);
    assertThat(index.metric()).isEqualTo(DistanceMetric.EUCLIDEAN);
    assertThat(index.size()).isEqualTo(0);
  }

  @Test
  @DisplayName("Create HNSW index succeeds with custom hyperparameters")
  void testCreateHnswIndex() throws IOException {
    Path indexFile = tempDir.resolve("test_hnsw.nvec");

    int exitCode =
        commandLine.execute(
            "create",
            indexFile.toString(),
            "--type",
            "HNSW",
            "--dimension",
            "8",
            "--metric",
            "COSINE",
            "--hnsw-m",
            "8",
            "--hnsw-m0",
            "16",
            "--hnsw-ef-construction",
            "64",
            "--hnsw-ef-search",
            "32");

    assertThat(exitCode).isEqualTo(0);
    assertThat(stderr()).isEmpty();
    assertThat(stdout()).contains("Successfully created HNSW index");

    VectorIndex index = NvecReader.read(indexFile);
    assertThat(index).isInstanceOf(HnswIndex.class);
    assertThat(index.dimension()).isEqualTo(8);
    assertThat(index.metric()).isEqualTo(DistanceMetric.COSINE);
    assertThat(index.size()).isEqualTo(0);

    // Verify inspect output for custom hyperparameters
    NvecInspectionResult result = NvecReader.inspect(indexFile);
    assertThat(result.isHnsw()).isTrue();
    assertThat(result.hnswMetadata().m()).isEqualTo(8);
    assertThat(result.hnswMetadata().m0()).isEqualTo(16);
    assertThat(result.hnswMetadata().efConstruction()).isEqualTo(64);
  }

  @Test
  @DisplayName("Create fails if target file already exists without --force")
  void testCreateTargetAlreadyExistsWithoutForce() throws IOException {
    Path indexFile = tempDir.resolve("existing.nvec");
    Files.writeString(indexFile, "dummy");

    int exitCode =
        commandLine.execute("create", indexFile.toString(), "--type", "FLAT", "--dimension", "4");

    assertThat(exitCode).isEqualTo(1);
    assertThat(stderr()).contains("Target file already exists").contains("--force");
  }

  @Test
  @DisplayName("Create succeeds and overwrites existing file when --force is provided")
  void testCreateTargetAlreadyExistsWithForce() throws IOException {
    Path indexFile = tempDir.resolve("existing_overwrite.nvec");
    Files.writeString(indexFile, "dummy");

    int exitCode =
        commandLine.execute(
            "create", indexFile.toString(), "--type", "FLAT", "--dimension", "4", "--force");

    assertThat(exitCode).isEqualTo(0);
    VectorIndex index = NvecReader.read(indexFile);
    assertThat(index.size()).isEqualTo(0);
  }

  @Test
  @DisplayName("Create fails on invalid dimension or unsupported metric")
  void testCreateInvalidParameters() {
    Path indexFile = tempDir.resolve("invalid.nvec");

    int exitCodeDim = commandLine.execute("create", indexFile.toString(), "--dimension", "-5");
    assertThat(exitCodeDim).isEqualTo(1);
    assertThat(stderr()).contains("Dimension must be positive");

    errStream.reset();
    int exitCodeMetric =
        commandLine.execute(
            "create", indexFile.toString(), "--dimension", "4", "--metric", "MANHATTAN");
    assertThat(exitCodeMetric).isEqualTo(1);
    assertThat(stderr()).contains("Unsupported distance metric 'MANHATTAN'");
  }

  // ═════════════════════════════════════════════════════════════════════
  // 2. INSERT COMMAND TESTS
  // ═════════════════════════════════════════════════════════════════════

  @Test
  @DisplayName("Insert single CSV batch successfully persists vectors into index")
  void testInsertSingleBatch() throws IOException {
    Path indexFile = tempDir.resolve("insert_single.nvec");
    commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "3", "-m", "EUCLIDEAN");

    Path csvFile = tempDir.resolve("vectors.csv");
    Files.writeString(
        csvFile,
        """
        id,vector
        1001,"0.1,0.2,0.3"
        1002,"0.4,0.5,0.6"
        1003,"0.7,0.8,0.9"
        """);

    outStream.reset();
    int exitCode =
        commandLine.execute("insert", indexFile.toString(), "--file", csvFile.toString());

    assertThat(exitCode).isEqualTo(0);
    assertThat(stderr()).isEmpty();
    assertThat(stdout())
        .contains("Successfully inserted 3 vectors")
        .contains("Total vector count: 3");

    // Verify index on disk
    FlatIndex restored = NvecReader.readFlat(indexFile);
    assertThat(restored.size()).isEqualTo(3);
    assertThat(restored.getVector(0)).containsExactly(0.1f, 0.2f, 0.3f);
    assertThat(restored.getVector(1)).containsExactly(0.4f, 0.5f, 0.6f);
    assertThat(restored.getVector(2)).containsExactly(0.7f, 0.8f, 0.9f);
  }

  @Test
  @DisplayName("Multiple incremental insert batches preserve all previous data")
  void testInsertMultipleBatchesPreservesOldData() throws IOException {
    Path indexFile = tempDir.resolve("incremental.nvec");
    commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2", "-m", "EUCLIDEAN");

    // Batch 1
    Path batch1 = tempDir.resolve("batch1.csv");
    Files.writeString(
        batch1,
        """
        id,vector
        1,"1.0,2.0"
        2,"3.0,4.0"
        """);
    int exit1 = commandLine.execute("insert", indexFile.toString(), "-f", batch1.toString());
    assertThat(exit1).isEqualTo(0);

    // Batch 2
    Path batch2 = tempDir.resolve("batch2.csv");
    Files.writeString(
        batch2,
        """
        id,vector
        3,"5.0,6.0"
        4,"7.0,8.0"
        """);
    int exit2 = commandLine.execute("insert", indexFile.toString(), "-f", batch2.toString());
    assertThat(exit2).isEqualTo(0);

    // Verify index has all 4 vectors intact
    FlatIndex restored = NvecReader.readFlat(indexFile);
    assertThat(restored.size()).isEqualTo(4);
    assertThat(restored.vectorData().getExternalId(0)).isEqualTo(1L);
    assertThat(restored.vectorData().getExternalId(1)).isEqualTo(2L);
    assertThat(restored.vectorData().getExternalId(2)).isEqualTo(3L);
    assertThat(restored.vectorData().getExternalId(3)).isEqualTo(4L);
    assertThat(restored.getVector(0)).containsExactly(1.0f, 2.0f);
    assertThat(restored.getVector(3)).containsExactly(7.0f, 8.0f);
  }

  @Test
  @DisplayName("Insert with duplicate ID within CSV fails atomically without modifying disk")
  void testInsertDuplicateIdInBatchFailsAtomically() throws IOException {
    Path indexFile = tempDir.resolve("dup_batch.nvec");
    commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");

    Path badCsv = tempDir.resolve("dup_ids.csv");
    Files.writeString(
        badCsv,
        """
        id,vector
        1001,"1.0,2.0"
        1002,"3.0,4.0"
        1001,"5.0,6.0"
        """);

    errStream.reset();
    int exitCode = commandLine.execute("insert", indexFile.toString(), "-f", badCsv.toString());

    assertThat(exitCode).isEqualTo(1);
    assertThat(stderr()).contains("Line 4").contains("Duplicate ID 1001");

    // Verify index file on disk remains empty (atomicity)
    VectorIndex index = NvecReader.read(indexFile);
    assertThat(index.size()).isEqualTo(0);
  }

  @Test
  @DisplayName("Insert with ID already existing in index fails atomically")
  void testInsertDuplicateIdAgainstExistingIndexFailsAtomically() throws IOException {
    Path indexFile = tempDir.resolve("dup_existing.nvec");
    commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");

    Path seedCsv = tempDir.resolve("seed.csv");
    Files.writeString(seedCsv, "id,vector\n1001,\"1.0,2.0\"\n");
    commandLine.execute("insert", indexFile.toString(), "-f", seedCsv.toString());

    Path conflictCsv = tempDir.resolve("conflict.csv");
    Files.writeString(
        conflictCsv,
        """
        id,vector
        2001,"3.0,4.0"
        1001,"5.0,6.0"
        """);

    errStream.reset();
    int exitCode =
        commandLine.execute("insert", indexFile.toString(), "-f", conflictCsv.toString());

    assertThat(exitCode).isEqualTo(1);
    assertThat(stderr()).contains("Line 3").contains("ID 1001 already exists in index");

    // Target file must retain ONLY original 1 vector (2001 was NOT inserted)
    VectorIndex index = NvecReader.read(indexFile);
    assertThat(index.size()).isEqualTo(1);
  }

  @Test
  @DisplayName("Insert with dimension mismatch fails atomically with line number")
  void testInsertDimensionMismatchFailsAtomically() throws IOException {
    Path indexFile = tempDir.resolve("dim_mismatch.nvec");
    commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "4");

    Path badCsv = tempDir.resolve("bad_dim.csv");
    Files.writeString(
        badCsv,
        """
        id,vector
        1,"0.1,0.2,0.3,0.4"
        2,"0.1,0.2,0.3"
        """);

    errStream.reset();
    int exitCode = commandLine.execute("insert", indexFile.toString(), "-f", badCsv.toString());

    assertThat(exitCode).isEqualTo(1);
    assertThat(stderr()).contains("Line 3").contains("Dimension mismatch (expected 4, got 3)");

    VectorIndex index = NvecReader.read(indexFile);
    assertThat(index.size()).isEqualTo(0);
  }

  @Test
  @DisplayName("Insert with NaN / Infinity float fails atomically with line number")
  void testInsertNonFiniteFloatFailsAtomically() throws IOException {
    Path indexFile = tempDir.resolve("nan_inf.nvec");
    commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");

    Path badCsv = tempDir.resolve("nan_vector.csv");
    Files.writeString(
        badCsv,
        """
        id,vector
        1,"1.0,NaN"
        """);

    errStream.reset();
    int exitCode = commandLine.execute("insert", indexFile.toString(), "-f", badCsv.toString());

    assertThat(exitCode).isEqualTo(1);
    assertThat(stderr()).contains("Line 2").contains("Non-finite float value");

    VectorIndex index = NvecReader.read(indexFile);
    assertThat(index.size()).isEqualTo(0);
  }

  @Test
  @DisplayName("Insert with malformed CSV format fails with clear line error")
  void testInsertMalformedRowFailsAtomically() throws IOException {
    Path indexFile = tempDir.resolve("malformed.nvec");
    commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");

    Path badCsv = tempDir.resolve("malformed.csv");
    Files.writeString(
        badCsv,
        """
        id,vector
        not_a_number,"1.0,2.0"
        """);

    errStream.reset();
    int exitCode = commandLine.execute("insert", indexFile.toString(), "-f", badCsv.toString());

    assertThat(exitCode).isEqualTo(1);
    assertThat(stderr()).contains("Line 2").contains("Invalid ID");

    VectorIndex index = NvecReader.read(indexFile);
    assertThat(index.size()).isEqualTo(0);
  }

  // ═════════════════════════════════════════════════════════════════════
  // 3. QUERY COMMAND TESTS
  // ═════════════════════════════════════════════════════════════════════

  @Test
  @DisplayName("Query against FLAT index returns exact k-NN ground truth results")
  void testQueryExactMatchFlatIndex() throws IOException {
    Path indexFile = tempDir.resolve("query_flat.nvec");
    commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2", "-m", "EUCLIDEAN");

    Path csv = tempDir.resolve("query_data.csv");
    Files.writeString(
        csv,
        """
        id,vector
        10,"1.0,0.0"
        20,"0.0,1.0"
        30,"10.0,10.0"
        """);
    commandLine.execute("insert", indexFile.toString(), "-f", csv.toString());

    outStream.reset();
    int exitCode =
        commandLine.execute("query", indexFile.toString(), "-k", "2", "--vector", "1.0,0.0");

    assertThat(exitCode).isEqualTo(0);
    assertThat(stderr()).isEmpty();
    String out = stdout();
    assertThat(out).contains("Query Results (Top-2");
    assertThat(out).contains("Rank").contains("External ID").contains("Distance");
    assertThat(out).contains("1        10                       0.000000");
    assertThat(out).contains("2        20");
    assertThat(out).doesNotContain("30"); // ID 30 is outside top-2
  }

  @Test
  @DisplayName("Query against HNSW index supports --ef-search parameter override")
  void testQueryHnswWithEfSearchOverride() throws IOException {
    Path indexFile = tempDir.resolve("query_hnsw.nvec");
    commandLine.execute(
        "create",
        indexFile.toString(),
        "-t",
        "HNSW",
        "-d",
        "3",
        "-m",
        "COSINE",
        "--hnsw-m",
        "8",
        "--hnsw-ef-construction",
        "64");

    Path csv = tempDir.resolve("hnsw_data.csv");
    Files.writeString(
        csv,
        """
        id,vector
        101,"0.1,0.2,0.3"
        102,"0.4,0.5,0.6"
        103,"0.7,0.8,0.9"
        """);
    commandLine.execute("insert", indexFile.toString(), "-f", csv.toString());

    outStream.reset();
    int exitCode =
        commandLine.execute(
            "query",
            indexFile.toString(),
            "-k",
            "2",
            "--vector",
            "0.1,0.2,0.3",
            "--ef-search",
            "50");

    assertThat(exitCode).isEqualTo(0);
    assertThat(stderr()).isEmpty();
    String out = stdout();
    assertThat(out).contains("Query Results (Top-2");
    assertThat(out).contains("1        101");
  }

  @Test
  @DisplayName("Query against empty index reports empty index cleanly")
  void testQueryEmptyIndex() {
    Path indexFile = tempDir.resolve("empty_query.nvec");
    commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");

    outStream.reset();
    int exitCode =
        commandLine.execute("query", indexFile.toString(), "-k", "5", "--vector", "1.0,2.0");

    assertThat(exitCode).isEqualTo(0);
    assertThat(stdout()).contains("Index is empty");
  }

  @Test
  @DisplayName("Query with vector dimension mismatch fails with exit code 1")
  void testQueryDimensionMismatchFails() {
    Path indexFile = tempDir.resolve("query_dim_mismatch.nvec");
    commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "4");

    int exitCode =
        commandLine.execute("query", indexFile.toString(), "-k", "5", "--vector", "1.0,2.0");

    assertThat(exitCode).isEqualTo(1);
    assertThat(stderr()).contains("dimension mismatch: expected 4, got 2");
  }

  // ═════════════════════════════════════════════════════════════════════
  // 4. HELP COMMAND TESTS
  // ═════════════════════════════════════════════════════════════════════

  @Test
  @DisplayName("Help subcommand displays detailed usage information for subcommands")
  void testHelpSubcommand() {
    int exitCodeCreate = commandLine.execute("help", "create");
    assertThat(exitCodeCreate).isEqualTo(0);
    assertThat(stdout()).contains("Create a new vector index").contains("--dimension");

    outStream.reset();
    int exitCodeInsert = commandLine.execute("help", "insert");
    assertThat(exitCodeInsert).isEqualTo(0);
    assertThat(stdout()).contains("Insert vectors from a CSV file").contains("--file");

    outStream.reset();
    int exitCodeQuery = commandLine.execute("help", "query");
    assertThat(exitCodeQuery).isEqualTo(0);
    assertThat(stdout()).contains("Query k-NN nearest vectors").contains("--vector");
  }
}
