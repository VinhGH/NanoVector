package com.nanovector.cli.command;

import com.nanovector.persistence.exception.CorruptIndexException;
import com.nanovector.persistence.exception.UnsupportedVersionException;
import com.nanovector.persistence.format.HnswMetadata;
import com.nanovector.persistence.reader.NvecInspectionResult;
import com.nanovector.persistence.reader.NvecReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.Callable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Spec;

/**
 * CLI command that inspects an NVEC v1 binary index file without loading vectors into memory.
 * Validates header invariants, prints format details, and verifies hardware-accelerated CRC32C.
 */
@Command(
    name = "inspect",
    mixinStandardHelpOptions = true,
    description = "Inspect binary .nvec index file header, metadata, and verify CRC32C integrity.")
public class InspectCommand implements Callable<Integer> {

  @Spec private CommandSpec spec;

  @Parameters(index = "0", description = "Path to .nvec file to inspect")
  private Path filePath;

  public InspectCommand() {}

  public InspectCommand(Path filePath) {
    this.filePath = filePath;
  }

  @Override
  public Integer call() {
    PrintWriter out =
        spec != null ? spec.commandLine().getOut() : new PrintWriter(System.out, true);
    PrintWriter err =
        spec != null ? spec.commandLine().getErr() : new PrintWriter(System.err, true);

    if (filePath == null) {
      err.println("Error: Missing file path argument.");
      return 1;
    }

    if (!Files.exists(filePath)) {
      err.println("Error: File not found: " + filePath);
      return 1;
    }

    try {
      NvecInspectionResult result = NvecReader.inspect(filePath);
      printInspectionReport(out, result);
      return 0;
    } catch (CorruptIndexException e) {
      err.println("Error: Corrupted index file: " + e.getMessage());
      return 1;
    } catch (UnsupportedVersionException e) {
      err.println("Error: Unsupported index version: " + e.getMessage());
      return 1;
    } catch (IOException e) {
      err.println("Error: Failed to read index file: " + e.getMessage());
      return 1;
    }
  }

  private void printInspectionReport(PrintWriter out, NvecInspectionResult result) {
    out.println("================================================================================");
    out.println("NVEC Binary Index Inspection Report");
    out.println("================================================================================");
    out.printf(Locale.ROOT, "%-22s: %s%n", "File Path", filePath.toAbsolutePath().normalize());
    out.printf(Locale.ROOT, "%-22s: %s%n", "File Size", formatFileSize(result.fileSizeBytes()));
    out.printf(Locale.ROOT, "%-22s: NVEC (0x4E564543)%n", "Magic Bytes");
    out.printf(Locale.ROOT, "%-22s: %d%n", "Format Version", result.header().version());
    out.printf(
        Locale.ROOT, "%-22s: Little Endian (0x%02X)%n", "Endianness", result.header().endianness());
    out.printf(Locale.ROOT, "%-22s: %s%n", "Index Type", result.header().indexType());
    out.printf(Locale.ROOT, "%-22s: %s%n", "Distance Metric", result.header().metric());
    out.printf(Locale.ROOT, "%-22s: %d%n", "Vector Dimension", result.header().dimension());
    out.printf(Locale.ROOT, "%-22s: %,d%n", "Vector Count", result.header().vectorCount());
    out.printf(Locale.ROOT, "%-22s: %d bytes%n", "Metadata Length", result.metadataLengthBytes());

    if (result.isHnsw()) {
      out.println(
          "--------------------------------------------------------------------------------");
      out.println("HNSW Topology Parameters:");
      HnswMetadata meta = result.hnswMetadata();
      out.printf(Locale.ROOT, "  %-20s: %d%n", "M (Max Outgoing)", meta.m());
      out.printf(Locale.ROOT, "  %-20s: %d%n", "M0 (Layer 0 Outgoing)", meta.m0());
      out.printf(Locale.ROOT, "  %-20s: %d%n", "efConstruction", meta.efConstruction());
      out.printf(Locale.ROOT, "  %-20s: %d%n", "defaultEfSearch", meta.defaultEfSearch());
      out.printf(Locale.ROOT, "  %-20s: %d%n", "Max Graph Level", meta.maxLevel());
      out.printf(Locale.ROOT, "  %-20s: %d%n", "Entry Node ID", meta.entryPointId());
    }

    out.println("--------------------------------------------------------------------------------");
    out.println("Integrity Check:");
    out.printf(Locale.ROOT, "  %-20s: 0x%08X%n", "Stored CRC32C", result.storedCrc32c());
    out.printf(Locale.ROOT, "  %-20s: 0x%08X%n", "Computed CRC32C", result.computedCrc32c());
    out.printf(
        Locale.ROOT,
        "  %-20s: %s%n",
        "Verification Status",
        result.crcValid() ? "VALID (CRC32C Match)" : "INVALID (Checksum Mismatch)");
    out.println("================================================================================");
    out.flush();
  }

  private String formatFileSize(long bytes) {
    if (bytes < 1024) {
      return bytes + " bytes";
    } else if (bytes < 1024 * 1024) {
      return String.format(Locale.ROOT, "%,d bytes (%.2f KB)", bytes, bytes / 1024.0);
    } else if (bytes < 1024L * 1024 * 1024) {
      return String.format(Locale.ROOT, "%,d bytes (%.2f MB)", bytes, bytes / (1024.0 * 1024));
    } else {
      return String.format(
          Locale.ROOT, "%,d bytes (%.2f GB)", bytes, bytes / (1024.0 * 1024 * 1024));
    }
  }
}
