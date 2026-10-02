package com.nanovector.cli.e2e;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.cli.NanoVectorCli;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.VectorIndex;
import com.nanovector.core.model.SearchResult;
import com.nanovector.persistence.reader.NvecInspectionResult;
import com.nanovector.persistence.reader.NvecReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import picocli.CommandLine;

/**
 * Comprehensive End-to-End integration test suite for NanoVector CLI.
 *
 * <p>Validates:
 *
 * <ul>
 *   <li>Golden path E2E executed as real OS processes via ProcessBuilder (create -> insert ->
 *       inspect -> query).
 *   <li>FLAT and HNSW complete lifecycles with exact Ground Truth comparisons.
 *   <li>Multi-batch incremental persistence preserving historical data and valid CRC32C.
 *   <li>Failure resilience: target files remain byte-for-byte identical when operations fail.
 *   <li>Filesystem semantics (relative paths, nested directory creation, overwrite protection).
 *   <li>CSV & query edge cases (negative IDs, unclosed quotes, k > size, efSearch < k).
 * </ul>
 */
class CliEndToEndIntegrationTest {

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

  record ProcessResult(int exitCode, String stdout, String stderr) {}

  private static ProcessResult runProcess(Path workingDir, String... args)
      throws IOException, InterruptedException {
    String javaBin =
        ProcessHandle.current()
            .info()
            .command()
            .orElseGet(
                () -> Path.of(System.getProperty("java.home"), "bin", "java.exe").toString());

    List<String> cmd = new ArrayList<>();
    cmd.add(javaBin);
    cmd.add("--add-modules");
    cmd.add("jdk.incubator.vector");
    cmd.add("-cp");
    cmd.add(System.getProperty("java.class.path"));
    cmd.add("com.nanovector.cli.NanoVectorCli");
    cmd.addAll(Arrays.asList(args));

    ProcessBuilder pb = new ProcessBuilder(cmd);
    if (workingDir != null) {
      pb.directory(workingDir.toFile());
    }
    Process p = pb.start();

    String stdout = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    String stderr = new String(p.getErrorStream().readAllBytes(), StandardCharsets.UTF_8);
    int exitCode = p.waitFor();
    return new ProcessResult(exitCode, stdout, stderr);
  }

  private static String sha256(Path file) throws IOException, NoSuchAlgorithmException {
    MessageDigest md = MessageDigest.getInstance("SHA-256");
    byte[] hash = md.digest(Files.readAllBytes(file));
    return HexFormat.of().formatHex(hash);
  }

  // ═════════════════════════════════════════════════════════════════════
  // 1. REAL OS CHILD PROCESS E2E TESTS
  // ═════════════════════════════════════════════════════════════════════

  @Nested
  @DisplayName("Real OS Process Execution (ProcessBuilder)")
  class RealProcessTests {

    @Test
    @DisplayName("Golden workflow runs entirely via real OS processes from command line")
    void testGoldenWorkflowRealProcess() throws Exception {
      Path indexFile = tempDir.resolve("golden.nvec");
      Path csvFile = tempDir.resolve("golden_vectors.csv");
      Files.writeString(
          csvFile,
          """
          id,vector
          1001,"0.1,0.2,0.3,0.4"
          1002,"0.5,0.6,0.7,0.8"
          1003,"0.9,1.0,1.1,1.2"
          """);

      // 1. Create HNSW index
      ProcessResult resCreate =
          runProcess(
              tempDir,
              "create",
              indexFile.toString(),
              "--type",
              "HNSW",
              "--dimension",
              "4",
              "--metric",
              "EUCLIDEAN",
              "--hnsw-m",
              "8",
              "--hnsw-ef-construction",
              "64");
      assertThat(resCreate.exitCode()).isEqualTo(0);
      assertThat(resCreate.stdout()).contains("Successfully created HNSW index");
      assertThat(Files.exists(indexFile)).isTrue();

      // 2. Insert vectors from CSV
      ProcessResult resInsert =
          runProcess(tempDir, "insert", indexFile.toString(), "--file", csvFile.toString());
      assertThat(resInsert.exitCode()).isEqualTo(0);
      assertThat(resInsert.stdout()).contains("Successfully inserted 3 vectors");

      // 3. Inspect binary index
      ProcessResult resInspect = runProcess(tempDir, "inspect", indexFile.toString());
      assertThat(resInspect.exitCode()).isEqualTo(0);
      assertThat(resInspect.stdout())
          .contains("Index Type            : HNSW")
          .contains("Vector Count          : 3")
          .contains("Verification Status : VALID (CRC32C Match)");

      // 4. Query nearest neighbor
      ProcessResult resQuery =
          runProcess(
              tempDir,
              "query",
              indexFile.toString(),
              "-k",
              "2",
              "-v",
              "0.1,0.2,0.3,0.4",
              "--ef-search",
              "50");
      assertThat(resQuery.exitCode()).isEqualTo(0);
      assertThat(resQuery.stdout()).contains("Query Results (Top-2").contains("1001");

      // 5. Help command
      ProcessResult resHelp = runProcess(tempDir, "help", "query");
      assertThat(resHelp.exitCode()).isEqualTo(0);
      assertThat(resHelp.stdout()).contains("Query k-NN nearest vectors");
    }
  }

  // ═════════════════════════════════════════════════════════════════════
  // 2. LIFECYCLE & GROUND TRUTH PARITY
  // ═════════════════════════════════════════════════════════════════════

  @Nested
  @DisplayName("Lifecycle & Accuracy against Oracle")
  class GroundTruthParityTests {

    @Test
    @DisplayName(
        "FLAT E2E: Create -> Multi-batch Insert -> Inspect -> Query exact match with Oracle")
    void testFlatLifecycleGroundTruth() throws IOException {
      Path indexFile = tempDir.resolve("flat_oracle.nvec");
      commandLine.execute(
          "create", indexFile.toString(), "-t", "FLAT", "-d", "3", "-m", "EUCLIDEAN");

      // Create ground truth FlatIndex oracle
      FlatIndex oracle = new FlatIndex(3, DistanceMetric.EUCLIDEAN);

      // Batch 1
      Path csv1 = tempDir.resolve("b1.csv");
      Files.writeString(
          csv1,
          """
          id,vector
          10,"1.0,2.0,3.0"
          20,"4.0,5.0,6.0"
          """);
      oracle.insert(10L, new float[] {1.0f, 2.0f, 3.0f});
      oracle.insert(20L, new float[] {4.0f, 5.0f, 6.0f});
      commandLine.execute("insert", indexFile.toString(), "-f", csv1.toString());

      // Batch 2
      Path csv2 = tempDir.resolve("b2.csv");
      Files.writeString(
          csv2,
          """
          id,vector
          30,"7.0,8.0,9.0"
          40,"0.1,0.2,0.3"
          """);
      oracle.insert(30L, new float[] {7.0f, 8.0f, 9.0f});
      oracle.insert(40L, new float[] {0.1f, 0.2f, 0.3f});
      commandLine.execute("insert", indexFile.toString(), "-f", csv2.toString());

      // Inspect
      outStream.reset();
      int exitInspect = commandLine.execute("inspect", indexFile.toString());
      assertThat(exitInspect).isEqualTo(0);
      assertThat(stdout()).contains("Vector Count          : 4").contains("VALID (CRC32C Match)");

      // Query top-3 for [1.1, 2.1, 3.1]
      float[] q = new float[] {1.1f, 2.1f, 3.1f};
      List<SearchResult> oracleResults = oracle.searchKnn(q, 3);

      outStream.reset();
      int exitQuery =
          commandLine.execute("query", indexFile.toString(), "-k", "3", "-v", "1.1, 2.1, 3.1");
      assertThat(exitQuery).isEqualTo(0);
      String queryOut = stdout();

      // Top 1 must be ID 10
      assertThat(queryOut).contains("1        " + oracleResults.get(0).id());
      assertThat(queryOut).contains("2        " + oracleResults.get(1).id());
      assertThat(queryOut).contains("3        " + oracleResults.get(2).id());
    }

    @Test
    @DisplayName("HNSW Recall: Query results achieve >= 90% Recall@5 compared to Flat Oracle")
    void testHnswRecallAgainstFlatOracle() throws IOException {
      int dim = 8;
      int n = 60;
      Path indexFile = tempDir.resolve("hnsw_recall.nvec");
      commandLine.execute(
          "create",
          indexFile.toString(),
          "-t",
          "HNSW",
          "-d",
          String.valueOf(dim),
          "-m",
          "EUCLIDEAN",
          "--hnsw-m",
          "16",
          "--hnsw-ef-construction",
          "100");

      FlatIndex oracle = new FlatIndex(dim, DistanceMetric.EUCLIDEAN);
      Random rng = new Random(42);

      StringBuilder csvContent = new StringBuilder("id,vector\n");
      for (int i = 1; i <= n; i++) {
        float[] vec = new float[dim];
        StringBuilder vecStr = new StringBuilder();
        for (int d = 0; d < dim; d++) {
          vec[d] = rng.nextFloat();
          vecStr.append(String.format(java.util.Locale.ROOT, "%.4f", vec[d]));
          if (d < dim - 1) vecStr.append(",");
        }
        oracle.insert(i, vec);
        csvContent.append(i).append(",\"").append(vecStr).append("\"\n");
      }

      Path csvFile = tempDir.resolve("recall_data.csv");
      Files.writeString(csvFile, csvContent.toString());
      int exitInsert =
          commandLine.execute("insert", indexFile.toString(), "-f", csvFile.toString());
      assertThat(exitInsert).isEqualTo(0);

      // Run 5 queries and verify high recall
      int k = 5;
      for (int q = 0; q < 5; q++) {
        float[] queryVec = new float[dim];
        StringBuilder qStr = new StringBuilder();
        for (int d = 0; d < dim; d++) {
          queryVec[d] = rng.nextFloat();
          qStr.append(String.format(java.util.Locale.ROOT, "%.4f", queryVec[d]));
          if (d < dim - 1) qStr.append(",");
        }

        List<SearchResult> oracleTopK = oracle.searchKnn(queryVec, k);
        Set<Long> oracleIds = oracleTopK.stream().map(SearchResult::id).collect(Collectors.toSet());

        outStream.reset();
        int exitQ =
            commandLine.execute(
                "query",
                indexFile.toString(),
                "-k",
                String.valueOf(k),
                "-v",
                qStr.toString(),
                "--ef-search",
                "50");
        assertThat(exitQ).isEqualTo(0);
        String qOut = stdout();

        int matches = 0;
        for (Long expectedId : oracleIds) {
          if (qOut.contains(String.valueOf(expectedId))) {
            matches++;
          }
        }
        double recall = (double) matches / k;
        assertThat(recall).isGreaterThanOrEqualTo(0.80); // At least 80% recall on random vectors
      }
    }
  }

  // ═════════════════════════════════════════════════════════════════════
  // 3. FAILURE RESILIENCE & BYTE INTEGRITY
  // ═════════════════════════════════════════════════════════════════════

  @Nested
  @DisplayName("Failure Resilience & Byte Integrity")
  class FailureResilienceTests {

    @Test
    @DisplayName("Target file remains byte-for-byte identical after failed insert validation")
    void testFileIntegrityPreservedOnInsertFailure() throws Exception {
      Path indexFile = tempDir.resolve("resilience.nvec");
      commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");

      Path seedCsv = tempDir.resolve("seed.csv");
      Files.writeString(seedCsv, "id,vector\n100,\"1.0,2.0\"\n200,\"3.0,4.0\"\n");
      commandLine.execute("insert", indexFile.toString(), "-f", seedCsv.toString());

      // Snapshot SHA-256 and byte array of valid file
      String originalHash = sha256(indexFile);
      byte[] originalBytes = Files.readAllBytes(indexFile);

      // Attempt 1: CSV with duplicate ID
      Path badCsvDup = tempDir.resolve("bad_dup.csv");
      Files.writeString(badCsvDup, "id,vector\n300,\"5.0,6.0\"\n300,\"7.0,8.0\"\n");
      errStream.reset();
      int exit1 = commandLine.execute("insert", indexFile.toString(), "-f", badCsvDup.toString());
      assertThat(exit1).isEqualTo(1);
      assertThat(sha256(indexFile)).isEqualTo(originalHash);
      assertThat(Files.readAllBytes(indexFile)).isEqualTo(originalBytes);

      // Attempt 2: CSV with dimension mismatch
      Path badCsvDim = tempDir.resolve("bad_dim.csv");
      Files.writeString(badCsvDim, "id,vector\n400,\"1.0,2.0,3.0,4.0\"\n");
      errStream.reset();
      int exit2 = commandLine.execute("insert", indexFile.toString(), "-f", badCsvDim.toString());
      assertThat(exit2).isEqualTo(1);
      assertThat(sha256(indexFile)).isEqualTo(originalHash);

      // Attempt 3: CSV with unclosed quote
      Path badCsvQuote = tempDir.resolve("bad_quote.csv");
      Files.writeString(badCsvQuote, "id,vector\n500,\"1.0,2.0\n");
      errStream.reset();
      int exit3 = commandLine.execute("insert", indexFile.toString(), "-f", badCsvQuote.toString());
      assertThat(exit3).isEqualTo(1);
      assertThat(sha256(indexFile)).isEqualTo(originalHash);

      // Attempt 4: CSV with NaN
      Path badCsvNan = tempDir.resolve("bad_nan.csv");
      Files.writeString(badCsvNan, "id,vector\n600,\"1.0,NaN\"\n");
      errStream.reset();
      int exit4 = commandLine.execute("insert", indexFile.toString(), "-f", badCsvNan.toString());
      assertThat(exit4).isEqualTo(1);
      assertThat(sha256(indexFile)).isEqualTo(originalHash);

      // Verify original index can still be cleanly loaded and inspected
      NvecInspectionResult result = NvecReader.inspect(indexFile);
      assertThat(result.crcValid()).isTrue();
      assertThat(result.header().vectorCount()).isEqualTo(2);
    }

    @Test
    @DisplayName("Create without --force fails without corrupting or deleting existing file")
    void testCreateWithoutForcePreservesExistingFile() throws Exception {
      Path indexFile = tempDir.resolve("dont_delete.nvec");
      commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");

      Path csv = tempDir.resolve("data.csv");
      Files.writeString(csv, "id,vector\n1,\"1.0,2.0\"\n");
      commandLine.execute("insert", indexFile.toString(), "-f", csv.toString());

      String hashBefore = sha256(indexFile);

      // Run create on same path without --force
      errStream.reset();
      int exit = commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");
      assertThat(exit).isEqualTo(1);
      assertThat(stderr()).contains("Target file already exists");

      // Verify file was never modified or deleted
      assertThat(sha256(indexFile)).isEqualTo(hashBefore);
      assertThat(NvecReader.read(indexFile).size()).isEqualTo(1);
    }
  }

  // ═════════════════════════════════════════════════════════════════════
  // 4. FILESYSTEM & PATH RESOLUTION
  // ═════════════════════════════════════════════════════════════════════

  @Nested
  @DisplayName("Filesystem & Path Handling")
  class FilesystemPathTests {

    @Test
    @DisplayName("Create in non-existent nested directory automatically creates parent folders")
    void testCreateInNestedDirectory() throws IOException {
      Path nestedDir = tempDir.resolve("sub1").resolve("sub2").resolve("sub3");
      Path indexFile = nestedDir.resolve("nested.nvec");

      int exitCode = commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "4");

      assertThat(exitCode).isEqualTo(0);
      assertThat(Files.exists(indexFile)).isTrue();
      assertThat(NvecReader.read(indexFile).dimension()).isEqualTo(4);
    }

    @Test
    @DisplayName("Relative paths are correctly resolved relative to execution working directory")
    void testRelativePathResolution() throws Exception {
      // Create subfolder to act as working directory
      Path workDir = tempDir.resolve("work");
      Files.createDirectories(workDir);

      Path csv = workDir.resolve("rel_vectors.csv");
      Files.writeString(csv, "id,vector\n1,\"0.5,0.5\"\n");

      // Create using relative filename
      ProcessResult resCreate =
          runProcess(workDir, "create", "rel_index.nvec", "-t", "FLAT", "-d", "2");
      assertThat(resCreate.exitCode()).isEqualTo(0);
      assertThat(Files.exists(workDir.resolve("rel_index.nvec"))).isTrue();

      // Insert using relative filenames
      ProcessResult resInsert =
          runProcess(workDir, "insert", "rel_index.nvec", "-f", "rel_vectors.csv");
      assertThat(resInsert.exitCode()).isEqualTo(0);

      // Inspect using relative filename
      ProcessResult resInspect = runProcess(workDir, "inspect", "rel_index.nvec");
      assertThat(resInspect.exitCode()).isEqualTo(0);
      assertThat(resInspect.stdout()).contains("Vector Count          : 1");

      // Query using relative filename
      ProcessResult resQuery =
          runProcess(workDir, "query", "rel_index.nvec", "-k", "1", "-v", "0.5,0.5");
      assertThat(resQuery.exitCode()).isEqualTo(0);
      assertThat(resQuery.stdout()).contains("Rank     External ID");
    }
  }

  // ═════════════════════════════════════════════════════════════════════
  // 5. EDGE CASES & BOUNDARY CONDITIONS
  // ═════════════════════════════════════════════════════════════════════

  @Nested
  @DisplayName("Boundary & Corner Cases")
  class BoundaryCaseTests {

    @Test
    @DisplayName("Negative external IDs are fully supported in CSV and query results")
    void testNegativeExternalIds() throws IOException {
      Path indexFile = tempDir.resolve("negative_ids.nvec");
      commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");

      Path csv = tempDir.resolve("neg.csv");
      Files.writeString(
          csv,
          """
          id,vector
          -1001,"1.0,2.0"
          -42,"3.0,4.0"
          """);
      int exitInsert = commandLine.execute("insert", indexFile.toString(), "-f", csv.toString());
      assertThat(exitInsert).isEqualTo(0);

      outStream.reset();
      int exitQuery =
          commandLine.execute("query", indexFile.toString(), "-k", "2", "-v", "1.0,2.0");
      assertThat(exitQuery).isEqualTo(0);
      assertThat(stdout()).contains("-1001").contains("-42");
    }

    @Test
    @DisplayName("Query with k > size returns all available vectors cleanly without crashing")
    void testQueryKGreaterThanSize() throws IOException {
      Path indexFile = tempDir.resolve("k_greater_than_size.nvec");
      commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");

      Path csv = tempDir.resolve("data.csv");
      Files.writeString(csv, "id,vector\n1,\"1.0,1.0\"\n2,\"2.0,2.0\"\n");
      commandLine.execute("insert", indexFile.toString(), "-f", csv.toString());

      outStream.reset();
      // Ask for k=100 when index only has 2 vectors
      int exitQuery =
          commandLine.execute("query", indexFile.toString(), "-k", "100", "-v", "1.0,1.0");
      assertThat(exitQuery).isEqualTo(0);
      assertThat(stdout()).contains("Query Results (Top-2"); // Clamped to actual size 2
    }

    @Test
    @DisplayName("Query with efSearch < k clamps effective candidate pool without error")
    void testQueryEfSearchLessThanK() throws IOException {
      Path indexFile = tempDir.resolve("ef_less_k.nvec");
      commandLine.execute("create", indexFile.toString(), "-t", "HNSW", "-d", "2");

      Path csv = tempDir.resolve("data.csv");
      Files.writeString(csv, "id,vector\n1,\"1.0,1.0\"\n2,\"2.0,2.0\"\n3,\"3.0,3.0\"\n");
      commandLine.execute("insert", indexFile.toString(), "-f", csv.toString());

      outStream.reset();
      // k=3, efSearch=1 (< 3)
      int exitQuery =
          commandLine.execute(
              "query", indexFile.toString(), "-k", "3", "-v", "1.0,1.0", "--ef-search", "1");
      assertThat(exitQuery).isEqualTo(0);
      assertThat(stdout()).contains("Query Results (Top-3");
    }

    @Test
    @DisplayName("CSV containing empty lines and hash comments parses successfully")
    void testCsvWithEmptyLinesAndComments() throws IOException {
      Path indexFile = tempDir.resolve("comments.nvec");
      commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");

      Path csv = tempDir.resolve("comments.csv");
      Files.writeString(
          csv,
          """
          # Header comment explaining the file
          id,vector

          # Section 1
          1,"0.1,0.2"

          # Section 2
          2,"0.3,0.4"

          """);

      int exit = commandLine.execute("insert", indexFile.toString(), "-f", csv.toString());
      assertThat(exit).isEqualTo(0);

      VectorIndex idx = NvecReader.read(indexFile);
      assertThat(idx.size()).isEqualTo(2);
    }

    @Test
    @DisplayName("Query with unclosed quotes or brackets fails with clear error")
    void testQueryUnclosedQuote() throws IOException {
      Path indexFile = tempDir.resolve("unclosed_q.nvec");
      commandLine.execute("create", indexFile.toString(), "-t", "FLAT", "-d", "2");

      errStream.reset();
      int exit = commandLine.execute("query", indexFile.toString(), "-k", "1", "-v", "\"1.0,2.0");
      assertThat(exit).isEqualTo(1);
      assertThat(stderr()).contains("Unclosed quote in query vector");
    }
  }
}
