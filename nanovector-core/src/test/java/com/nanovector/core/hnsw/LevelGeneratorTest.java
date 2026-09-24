package com.nanovector.core.hnsw;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.offset;

class LevelGeneratorTest {

    @Test
    @DisplayName("LevelGenerator: Distribution decays exponentially with ratio ~ 1/M")
    void testExponentialLevelDistribution() {
        int m = 16;
        HnswConfig config = new HnswConfig(m, 32, 100, 50, 1.0 / Math.log(m), 42L);
        LevelGenerator generator = new LevelGenerator(config);

        int sampleCount = 200_000;
        int maxObservedLevel = 0;
        int[] countsAtOrAbove = new int[10];

        for (int i = 0; i < sampleCount; i++) {
            int level = generator.nextLevel();
            assertThat(level).isGreaterThanOrEqualTo(0);
            if (level > maxObservedLevel) {
                maxObservedLevel = level;
            }
            for (int l = 0; l <= Math.min(level, countsAtOrAbove.length - 1); l++) {
                countsAtOrAbove[l]++;
            }
        }

        // 100% of nodes must exist at or above level 0
        assertThat(countsAtOrAbove[0]).isEqualTo(sampleCount);

        // Theoretical decay factor between successive levels is 1 / M = 1 / 16 = 0.0625
        double expectedRatio = 1.0 / m;

        // Ratio P(level >= 1) / P(level >= 0)
        double ratio0to1 = (double) countsAtOrAbove[1] / countsAtOrAbove[0];
        assertThat(ratio0to1).isCloseTo(expectedRatio, offset(0.01));

        // Ratio P(level >= 2) / P(level >= 1)
        double ratio1to2 = (double) countsAtOrAbove[2] / countsAtOrAbove[1];
        assertThat(ratio1to2).isCloseTo(expectedRatio, offset(0.015));
    }

    @Test
    @DisplayName("LevelGenerator: Deterministic seed produces identical level sequences")
    void testDeterministicLevelGeneration() {
        long seed = 987654321L;
        LevelGenerator genA = new LevelGenerator(1.0 / Math.log(16), seed);
        LevelGenerator genB = new LevelGenerator(1.0 / Math.log(16), seed);
        LevelGenerator genDiff = new LevelGenerator(1.0 / Math.log(16), seed + 1);

        int count = 1000;
        int identicalCountDiff = 0;

        for (int i = 0; i < count; i++) {
            int levelA = genA.nextLevel();
            int levelB = genB.nextLevel();
            int levelDiff = genDiff.nextLevel();

            assertThat(levelA).isEqualTo(levelB);
            if (levelA == levelDiff) {
                identicalCountDiff++;
            }
        }

        // Generator with different seed must not be identical for all samples
        assertThat(identicalCountDiff).isLessThan(count);
    }

    @Test
    @DisplayName("HnswConfig: Parameter validation guards against illegal values")
    void testConfigValidation() {
        assertThatThrownBy(() -> new HnswConfig(1, 32, 100, 50, 0.5, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("m must be at least 2");

        assertThatThrownBy(() -> new HnswConfig(16, 15, 100, 50, 0.5, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("m0 must be at least m");

        assertThatThrownBy(() -> new HnswConfig(16, 32, 0, 50, 0.5, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("efConstruction must be positive");

        assertThatThrownBy(() -> new HnswConfig(16, 32, 100, 0, 0.5, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("efSearch must be positive");

        assertThatThrownBy(() -> new HnswConfig(16, 32, 100, 50, -1.0, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("mL must be a positive finite number");
    }
}
