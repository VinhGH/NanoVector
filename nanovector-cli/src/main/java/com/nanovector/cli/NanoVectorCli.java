package com.nanovector.cli;

import com.nanovector.cli.command.InspectCommand;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;

/**
 * NanoVector CLI root entry point.
 *
 * <p>Provides commands for vector index inspection, lifecycle operations, ingestion, and
 * high-performance querying without Spring Boot framework dependencies.
 */
@Command(
    name = "nanovector",
    mixinStandardHelpOptions = true,
    version = "NanoVector 0.1.0",
    description = "NanoVector CLI - Vector Similarity Search Engine Command Line Interface",
    subcommands = {InspectCommand.class})
public class NanoVectorCli implements Callable<Integer> {

  @Override
  public Integer call() {
    CommandLine.usage(this, System.out);
    return 0;
  }

  /**
   * Main entry point for standalone CLI execution.
   *
   * @param args command-line arguments
   */
  public static void main(String[] args) {
    int exitCode = execute(args);
    System.exit(exitCode);
  }

  /**
   * Executes CLI commands within the current JVM process.
   *
   * @param args command-line arguments
   * @return process exit code (0 for success, non-zero for failure)
   */
  public static int execute(String... args) {
    return createCommandLine().execute(args);
  }

  /**
   * Creates a preconfigured {@link CommandLine} instance for command execution or testing.
   *
   * @return configured CommandLine instance
   */
  public static CommandLine createCommandLine() {
    return new CommandLine(new NanoVectorCli());
  }
}
