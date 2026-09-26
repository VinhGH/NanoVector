package com.nanovector.benchmark.distance;

import com.nanovector.core.distance.DistanceCalculator;
import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.util.VectorUtils;
import java.util.Random;
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
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * JMH Benchmark comparing pure Scalar vs SIMD (Java Vector API) vector distance calculations.
 *
 * <p>Key experimental controls:
 *
 * <ul>
 *   <li>Same input vectors (vectorA, vectorB) evaluated by both implementations.
 *   <li>Data generation and normalization conducted strictly in {@link #setup()} outside timed
 *       methods.
 *   <li>Strict semantic equivalence assertion verified prior to timing execution.
 *   <li>Zero allocations in the timed benchmark path.
 * </ul>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 3, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(
    value = 1,
    jvmArgsAppend = {"--add-modules", "jdk.incubator.vector"})
public class ScalarVsSimdDistanceBenchmark {

  @Param({"EUCLIDEAN", "COSINE", "DOT_PRODUCT"})
  private DistanceMetric metric;

  @Param({"32", "64", "128", "384", "768", "1536"})
  private int dimension;

  private DistanceCalculator scalarCalc;
  private DistanceCalculator simdCalc;
  private float[] vectorA;
  private float[] vectorB;

  @Setup
  public void setup() {
    this.scalarCalc = DistanceCalculator.create(metric, false);
    this.simdCalc = DistanceCalculator.create(metric, true);

    this.vectorA = new float[dimension];
    this.vectorB = new float[dimension];

    Random rng = new Random(42L + dimension + metric.ordinal());
    for (int i = 0; i < dimension; i++) {
      this.vectorA[i] = rng.nextFloat() * 2.0f - 1.0f;
      this.vectorB[i] = rng.nextFloat() * 2.0f - 1.0f;
    }

    if (metric == DistanceMetric.COSINE) {
      this.vectorA = VectorUtils.normalize(this.vectorA);
      this.vectorB = VectorUtils.normalize(this.vectorB);
    }

    // Verify numerical equivalence within tolerance
    float scalarDist = scalarCalc.distance(vectorA, vectorB);
    float simdDist = simdCalc.distance(vectorA, vectorB);
    float diff = Math.abs(scalarDist - simdDist);
    float tolerance = (metric == DistanceMetric.COSINE) ? 1e-4f : 1e-3f;
    if (diff > tolerance) {
      throw new IllegalStateException(
          String.format(
              "Equivalence check failed for metric=%s, dimension=%d: scalar=%f, simd=%f, diff=%f",
              metric, dimension, scalarDist, simdDist, diff));
    }
  }

  @Benchmark
  public float scalarDistance() {
    return scalarCalc.distance(vectorA, vectorB);
  }

  @Benchmark
  public float simdDistance() {
    return simdCalc.distance(vectorA, vectorB);
  }

  // Package-private accessors for unit test verification
  void initForTest(DistanceMetric metric, int dimension) {
    this.metric = metric;
    this.dimension = dimension;
    setup();
  }

  DistanceCalculator scalarCalc() {
    return scalarCalc;
  }

  DistanceCalculator simdCalc() {
    return simdCalc;
  }

  public static void main(String[] args) throws RunnerException {
    Options opt =
        new OptionsBuilder()
            .include(ScalarVsSimdDistanceBenchmark.class.getSimpleName())
            .forks(1)
            .build();
    new Runner(opt).run();
  }
}
