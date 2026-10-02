package com.nanovector.cli.command;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.cli.NanoVectorCli;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.hnsw.HnswConfig;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.HnswIndex;
import com.nanovector.persistence.writer.NvecWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.CRC32C;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import picocli.CommandLine;

class InspectCommandTest {

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

  @Test
  @DisplayName("Inspect valid FLAT index succeeds with exit code 0 and accurate metadata")
  void testInspectValidFlatIndex() throws IOException {
    int dim = 8;
    FlatIndex index = new FlatIndex(dim, DistanceMetric.COSINE);
    index.insert(10L, new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
    index.insert(20L, new float[] {0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f});

    Path file = tempDir.resolve("valid_flat.nvec");
    NvecWriter.write(index, file);

    int exitCode = commandLine.execute("inspect", file.toString());

    assertThat(exitCode).isEqualTo(0);
    assertThat(stderr()).isEmpty();
    String out = stdout();
    assertThat(out).contains("NVEC Binary Index Inspection Report");
    assertThat(out).contains("Index Type            : FLAT");
    assertThat(out).contains("Distance Metric       : COSINE");
    assertThat(out).contains("Vector Dimension      : 8");
    assertThat(out).contains("Vector Count          : 2");
    assertThat(out).contains("Verification Status : VALID (CRC32C Match)");
    assertThat(out).doesNotContain("HNSW Topology Parameters");
  }

  @Test
  @DisplayName("Inspect valid HNSW index succeeds with exit code 0 and topology parameters")
  void testInspectValidHnswIndex() throws IOException {
    int dim = 4;
    HnswConfig config = new HnswConfig(8, 16, 64, 50, 1.0 / Math.log(8), 42L);
    HnswIndex index = new HnswIndex(dim, DistanceMetric.EUCLIDEAN, config);
    index.insert(1L, new float[] {1.0f, 2.0f, 3.0f, 4.0f});
    index.insert(2L, new float[] {5.0f, 6.0f, 7.0f, 8.0f});

    Path file = tempDir.resolve("valid_hnsw.nvec");
    NvecWriter.write(index, file);

    int exitCode = commandLine.execute("inspect", file.toString());

    assertThat(exitCode).isEqualTo(0);
    assertThat(stderr()).isEmpty();
    String out = stdout();
    assertThat(out).contains("Index Type            : HNSW");
    assertThat(out).contains("Distance Metric       : EUCLIDEAN");
    assertThat(out).contains("Vector Dimension      : 4");
    assertThat(out).contains("Vector Count          : 2");
    assertThat(out).contains("HNSW Topology Parameters:");
    assertThat(out).contains("M (Max Outgoing)    : 8");
    assertThat(out).contains("M0 (Layer 0 Outgoing): 16");
    assertThat(out).contains("efConstruction      : 64");
    assertThat(out).contains("Verification Status : VALID (CRC32C Match)");
  }

  @Test
  @DisplayName("Inspect corrupted CRC32C (bitflip) fails with exit code 1 and error message")
  void testInspectCorruptedCrc32c() throws IOException {
    FlatIndex index = new FlatIndex(4, DistanceMetric.EUCLIDEAN);
    index.insert(1L, new float[] {1.0f, 2.0f, 3.0f, 4.0f});

    Path file = tempDir.resolve("corrupted_crc.nvec");
    NvecWriter.write(index, file);

    // Corrupt a payload byte without updating footer CRC
    byte[] data = Files.readAllBytes(file);
    data[34] ^= 0xFF;
    Files.write(file, data);

    int exitCode = commandLine.execute("inspect", file.toString());

    assertThat(exitCode).isNotZero();
    assertThat(stderr()).contains("Corrupted index file").contains("CRC32C checksum mismatch");
  }

  @Test
  @DisplayName("Inspect file with bad magic bytes fails with exit code 1")
  void testInspectBadMagic() throws IOException {
    FlatIndex index = new FlatIndex(4, DistanceMetric.EUCLIDEAN);
    index.insert(1L, new float[] {1.0f, 2.0f, 3.0f, 4.0f});

    Path file = tempDir.resolve("bad_magic.nvec");
    NvecWriter.write(index, file);

    byte[] data = Files.readAllBytes(file);
    data[0] = 'B';
    data[1] = 'A';
    data[2] = 'D';
    data[3] = '!';
    Files.write(file, data);

    int exitCode = commandLine.execute("inspect", file.toString());

    assertThat(exitCode).isNotZero();
    assertThat(stderr()).contains("Corrupted index file").contains("magic");
  }

  @Test
  @DisplayName("Inspect file with unsupported version fails with exit code 1")
  void testInspectUnsupportedVersion() throws IOException {
    ByteBuffer buf = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    buf.put(new byte[] {'N', 'V', 'E', 'C'});
    buf.putShort((short) 99); // Unsupported version 99
    buf.put((byte) 1);
    buf.put((byte) 1);
    buf.put((byte) 1);
    buf.put(new byte[] {0, 0, 0});
    buf.putInt(4);
    buf.putInt(0);
    buf.put(new byte[12]);
    buf.putInt(0); // meta len
    CRC32C crc = new CRC32C();
    crc.update(buf.array(), 0, 36);
    buf.position(36);
    buf.putInt((int) crc.getValue());

    byte[] validBytes = new byte[40];
    System.arraycopy(buf.array(), 0, validBytes, 0, 40);

    Path file = tempDir.resolve("unsupported_version.nvec");
    Files.write(file, validBytes);

    int exitCode = commandLine.execute("inspect", file.toString());

    assertThat(exitCode).isNotZero();
    assertThat(stderr()).contains("Unsupported index version").contains("99");
  }

  @Test
  @DisplayName("Inspect truncated file (< 40 bytes) fails with exit code 1")
  void testInspectTruncatedFile() throws IOException {
    Path file = tempDir.resolve("truncated.nvec");
    Files.write(file, new byte[] {'N', 'V', 'E', 'C', 1, 0});

    int exitCode = commandLine.execute("inspect", file.toString());

    assertThat(exitCode).isNotZero();
    assertThat(stderr()).contains("Corrupted index file").contains("smaller than minimum");
  }

  @Test
  @DisplayName("Inspect non-existent file fails with exit code 1")
  void testInspectNonExistentFile() {
    Path nonExistent = tempDir.resolve("non_existent_file.nvec");

    int exitCode = commandLine.execute("inspect", nonExistent.toString());

    assertThat(exitCode).isNotZero();
    assertThat(stderr()).contains("File not found");
  }

  @Test
  @DisplayName("CLI help option displays command description and subcommands")
  void testCliHelp() {
    int exitCode = commandLine.execute("--help");

    assertThat(exitCode).isEqualTo(0);
    assertThat(stdout()).contains("nanovector").contains("inspect");
  }

  @Test
  @DisplayName("CLI version option displays version information")
  void testCliVersion() {
    int exitCode = commandLine.execute("--version");

    assertThat(exitCode).isEqualTo(0);
    assertThat(stdout()).contains("NanoVector 0.1.0");
  }
}
