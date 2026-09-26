package com.nanovector.benchmark;

import com.nanovector.core.distance.DistanceCalculator;
import com.nanovector.core.distance.DistanceMetric;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * JMH Smoke Benchmark verifying foundational harness setup for Phase 5.
 *
 * <p>Validates:
 *
 * <ul>
 *   <li>Maven module and annotation processor compilation.
 *   <li>JMH Fork, Warmup, and Measurement lifecycle execution.
 *   <li>Blackhole consumption preventing dead-code elimination.
 *   <li>Integration with {@code nanovector-core} using Vector API incubator module.
 * </ul>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 3, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Fork(
    value = 1,
    jvmArgsAppend = {"--add-modules", "jdk.incubator.vector"})
public class JmhSmokeBenchmark {

  @Param({"128"})
  private int dimension;

  private DistanceCalculator calculator;
  private float[] vectorA;
  private float[] vectorB;

  @Setup
  public void setup() {
    this.calculator = DistanceCalculator.create(DistanceMetric.EUCLIDEAN, true);
    this.vectorA = new float[dimension];
    this.vectorB = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      this.vectorA[i] = (float) Math.sin(i);
      this.vectorB[i] = (float) Math.cos(i);
    }
  }

  @Benchmark
  public void smokeDistanceCalculation(Blackhole bh) {
    float distance = calculator.distance(vectorA, 0, vectorB);
    bh.consume(distance);
  }

  public static void main(String[] args) throws RunnerException {
    Options opt =
        new OptionsBuilder().include(JmhSmokeBenchmark.class.getSimpleName()).forks(1).build();
    new Runner(opt).run();
  }
}
