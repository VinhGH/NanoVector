package com.nanovector.core.hnsw;

/**
 * Immutable configuration parameters for the HNSW index.
 *
 * <p>Hyperparameters adhere to the paper by Malkov &amp; Yashunin (2018):
 *
 * <ul>
 *   <li>{@code m}: Maximum number of outgoing edges per node at layers &gt; 0 (default: 16).
 *   <li>{@code m0}: Maximum number of outgoing edges per node at layer 0 (default: 2 * m = 32).
 *   <li>{@code efConstruction}: Size of dynamic candidate list during graph construction (default:
 *       100).
 *   <li>{@code efSearch}: Size of dynamic candidate list during nearest-neighbor search (default:
 *       50).
 *   <li>{@code mL}: Normalization factor for exponential level generation, typically {@code 1.0 /
 *       ln(m)}.
 *   <li>{@code seed}: Optional random seed for deterministic level generation and testing (null for
 *       non-deterministic).
 * </ul>
 */
public record HnswConfig(int m, int m0, int efConstruction, int efSearch, double mL, Long seed) {

  public static final int DEFAULT_M = 16;
  public static final int DEFAULT_M0 = 32;
  public static final int DEFAULT_EF_CONSTRUCTION = 100;
  public static final int DEFAULT_EF_SEARCH = 50;

  public HnswConfig {
    if (m < 2) {
      throw new IllegalArgumentException("m must be at least 2, but got: " + m);
    }
    if (m0 < m) {
      throw new IllegalArgumentException("m0 must be at least m (" + m + "), but got: " + m0);
    }
    if (efConstruction <= 0) {
      throw new IllegalArgumentException(
          "efConstruction must be positive, but got: " + efConstruction);
    }
    if (efSearch <= 0) {
      throw new IllegalArgumentException("efSearch must be positive, but got: " + efSearch);
    }
    if (mL <= 0.0 || Double.isNaN(mL) || Double.isInfinite(mL)) {
      throw new IllegalArgumentException("mL must be a positive finite number, but got: " + mL);
    }
  }

  /**
   * Creates a default configuration with M = 16, M0 = 32, efConstruction = 100, efSearch = 50, mL =
   * 1 / ln(16), and non-deterministic random generation.
   */
  public static HnswConfig defaultConfig() {
    return new HnswConfig(
        DEFAULT_M,
        DEFAULT_M0,
        DEFAULT_EF_CONSTRUCTION,
        DEFAULT_EF_SEARCH,
        1.0 / Math.log(DEFAULT_M),
        null);
  }

  /** Creates a configuration with default parameters and a fixed seed for deterministic testing. */
  public static HnswConfig withSeed(long seed) {
    return new HnswConfig(
        DEFAULT_M,
        DEFAULT_M0,
        DEFAULT_EF_CONSTRUCTION,
        DEFAULT_EF_SEARCH,
        1.0 / Math.log(DEFAULT_M),
        seed);
  }

  /** Returns a copy of this configuration with a modified {@code efSearch} parameter. */
  public HnswConfig withEfSearch(int newEfSearch) {
    return new HnswConfig(m, m0, efConstruction, newEfSearch, mL, seed);
  }

  /** Returns a copy of this configuration with a modified {@code efConstruction} parameter. */
  public HnswConfig withEfConstruction(int newEfConstruction) {
    return new HnswConfig(m, m0, newEfConstruction, efSearch, mL, seed);
  }
}
