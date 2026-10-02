package com.nanovector.server.service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Resolves and sanitizes file paths within the server's designated data storage directory.
 *
 * <p>Prevents arbitrary filesystem writes and path traversal vulnerabilities by enforcing strict
 * whitelist validation and path normalization.
 */
public final class StoragePathResolver {

  private static final Pattern SAFE_FILENAME_PATTERN = Pattern.compile("^[a-zA-Z0-9._-]+\\.nvec$");

  private final Path baseDir;

  public StoragePathResolver(Path baseDir) {
    Objects.requireNonNull(baseDir, "baseDir must not be null");
    this.baseDir = baseDir.toAbsolutePath().normalize();
  }

  public StoragePathResolver(String baseDirPath) {
    this(Paths.get(baseDirPath != null && !baseDirPath.isBlank() ? baseDirPath : "data/indexes"));
  }

  /**
   * Resolves a sanitized, safe {@link Path} for an index file inside the base directory.
   *
   * @param fileName desired file name or index identifier
   * @return normalized, verified Path inside base directory
   * @throws IllegalArgumentException if the file name contains path traversal or invalid characters
   */
  public Path resolveSafe(String fileName) {
    if (fileName == null || fileName.isBlank()) {
      throw new IllegalArgumentException("File name must not be null or blank");
    }

    String trimmed = fileName.trim();
    if (trimmed.contains("/") || trimmed.contains("\\") || trimmed.contains("..")) {
      throw new IllegalArgumentException(
          "Path traversal forbidden: file name contains path separators or '..'");
    }

    String effectiveName = trimmed.endsWith(".nvec") ? trimmed : trimmed + ".nvec";
    if (!SAFE_FILENAME_PATTERN.matcher(effectiveName).matches()) {
      throw new IllegalArgumentException(
          "Invalid file name '"
              + fileName
              + "'. Only alphanumeric characters, dashes, underscores, and dots are permitted.");
    }

    Path resolved = baseDir.resolve(effectiveName).normalize();
    if (!resolved.startsWith(baseDir)) {
      throw new IllegalArgumentException(
          "Path traversal violation: resolved path escapes base data directory");
    }

    return resolved;
  }

  /** Ensures that the base storage directory exists on disk. */
  public Path ensureBaseDirExists() throws IOException {
    if (!Files.exists(baseDir)) {
      Files.createDirectories(baseDir);
    }
    return baseDir;
  }

  /** Returns the normalized base directory path. */
  public Path baseDir() {
    return baseDir;
  }
}
