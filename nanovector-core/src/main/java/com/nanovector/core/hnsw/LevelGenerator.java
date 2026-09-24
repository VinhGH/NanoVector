package com.nanovector.core.hnsw;

import java.util.Objects;
import java.util.Random;

/**
 * Generates random node levels using the exponential decay distribution:
 *
 * <pre>
 *   level = floor(-ln(unif(0, 1)) * mL)
 * </pre>
 *
 * where {@code mL = 1.0 / ln(M)}.
 *
 * <p>Ensures that all nodes have level &ge; 0 (exist on layer 0), while higher layers have
 * exponentially fewer nodes to form the hierarchical skip-list routing graph.
 */
public final class LevelGenerator {

  private final double mL;
  private final Random random;

  public LevelGenerator(HnswConfig config) {
    Objects.requireNonNull(config, "HnswConfig must not be null");
    this.mL = config.mL();
    this.random = config.seed() != null ? new Random(config.seed()) : new Random();
  }

  public LevelGenerator(double mL, Long seed) {
    if (mL <= 0.0 || Double.isNaN(mL) || Double.isInfinite(mL)) {
      throw new IllegalArgumentException("mL must be positive and finite, but got: " + mL);
    }
    this.mL = mL;
    this.random = seed != null ? new Random(seed) : new Random();
  }

  /**
   * Draws the next random level for an inserted node.
   *
   * @return non-negative integer level (0, 1, 2, ...)
   */
  public int nextLevel() {
    double unif = random.nextDouble();
    // Guard against unif == 0.0 to prevent ln(0) = -Infinity
    while (unif == 0.0) {
      unif = random.nextDouble();
    }
    int level = (int) Math.floor(-Math.log(unif) * mL);
    return Math.max(0, level);
  }

  public double mL() {
    return mL;
  }
}
