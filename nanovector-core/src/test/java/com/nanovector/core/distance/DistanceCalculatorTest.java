package com.nanovector.core.distance;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.offset;

class DistanceCalculatorTest {

    @Test
    @DisplayName("Euclidean: Pythagoras 3-4-5 should yield 25.0f (squared)")
    void testEuclideanPythagoras() {
        DistanceCalculator calc = new ScalarEuclideanDistance();
        float[] a = {0.0f, 0.0f};
        float[] b = {3.0f, 4.0f};

        float dist = calc.distance(a, b);
        assertThat(dist).isEqualTo(25.0f);

        // Identical vectors must yield 0.0
        assertThat(calc.distance(a, a)).isEqualTo(0.0f);
    }

    @Test
    @DisplayName("Euclidean: Contiguous buffer offset must match standalone vector distance")
    void testEuclideanBufferOffset() {
        DistanceCalculator calc = new ScalarEuclideanDistance();
        // Buffer holding 3 vectors of dimension 3
        float[] buffer = {
                1.0f, 2.0f, 3.0f,  // Node 0 (offset 0)
                4.0f, 5.0f, 6.0f,  // Node 1 (offset 3)
                7.0f, 8.0f, 9.0f   // Node 2 (offset 6)
        };
        float[] query = {4.0f, 5.0f, 6.0f};

        // Node 1 matches query exactly -> dist = 0.0
        float distNode1 = calc.distance(buffer, 3, query);
        assertThat(distNode1).isEqualTo(0.0f);

        // Node 0 vs query -> (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27.0
        float distNode0 = calc.distance(buffer, 0, query);
        assertThat(distNode0).isEqualTo(27.0f);
    }

    @Test
    @DisplayName("Cosine: Collinear vectors must yield 0.0f distance")
    void testCosineCollinearVectors() {
        DistanceCalculator calc = new ScalarCosineDistance();
        // (3, 4) and (6, 8) normalized both become (0.6, 0.8)
        float[] u = {0.6f, 0.8f};
        float[] v = {0.6f, 0.8f};

        float dist = calc.distance(u, v);
        assertThat(dist).isCloseTo(0.0f, offset(1e-6f));
    }

    @Test
    @DisplayName("Cosine: Orthogonal vectors must yield 1.0f, opposite vectors must yield 2.0f")
    void testCosineOrthogonalAndOpposite() {
        DistanceCalculator calc = new ScalarCosineDistance();
        float[] right = {1.0f, 0.0f};
        float[] up = {0.0f, 1.0f};
        float[] left = {-1.0f, 0.0f};

        assertThat(calc.distance(right, up)).isCloseTo(1.0f, offset(1e-6f));
        assertThat(calc.distance(right, left)).isCloseTo(2.0f, offset(1e-6f));
    }

    @Test
    @DisplayName("DotProduct: Negated values adhere to distance minimization")
    void testDotProductDistanceMinimization() {
        DistanceCalculator calc = new ScalarDotProductDistance();
        float[] a = {1.0f, 0.0f};
        float[] b = {1.0f, 0.0f};
        float[] c = {0.0f, 1.0f};

        // a . b = 1.0 -> dist = -1.0 (more similar)
        // a . c = 0.0 -> dist = 0.0 (less similar)
        float distSame = calc.distance(a, b);
        float distOrtho = calc.distance(a, c);

        assertThat(distSame).isEqualTo(-1.0f);
        assertThat(distOrtho).isEqualTo(0.0f);
        assertThat(distSame).isLessThan(distOrtho);
    }

    @Test
    @DisplayName("Dimension mismatch must throw IllegalArgumentException")
    void testDimensionMismatch() {
        DistanceCalculator calc = new ScalarEuclideanDistance();
        float[] a = {1.0f, 2.0f};
        float[] b = {1.0f, 2.0f, 3.0f};

        assertThatThrownBy(() -> calc.distance(a, b))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("mismatch");
    }
}
