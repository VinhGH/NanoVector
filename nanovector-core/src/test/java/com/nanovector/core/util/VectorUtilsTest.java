package com.nanovector.core.util;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.offset;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class VectorUtilsTest {

  @Test
  @DisplayName("Normalize: (3, 4) converts to (0.6, 0.8) with unit norm")
  void testNormalizeVector() {
    float[] v = {3.0f, 4.0f};
    float[] norm = VectorUtils.normalize(v);

    assertThat(norm[0]).isCloseTo(0.6f, offset(1e-6f));
    assertThat(norm[1]).isCloseTo(0.8f, offset(1e-6f));
    assertThat(VectorUtils.norm(norm)).isCloseTo(1.0f, offset(1e-6f));
  }

  @Test
  @DisplayName("Normalize: Collinear vectors (3, 4) and (6, 8) produce identical unit vectors")
  void testCollinearNormalization() {
    float[] v1 = {3.0f, 4.0f};
    float[] v2 = {6.0f, 8.0f};

    float[] n1 = VectorUtils.normalize(v1);
    float[] n2 = VectorUtils.normalize(v2);

    assertThat(n1[0]).isCloseTo(n2[0], offset(1e-6f));
    assertThat(n1[1]).isCloseTo(n2[1], offset(1e-6f));
  }

  @Test
  @DisplayName("Normalize: Zero vector must throw IllegalArgumentException (no silent NaN)")
  void testZeroVectorThrowsException() {
    float[] zero = {0.0f, 0.0f, 0.0f};

    assertThatThrownBy(() -> VectorUtils.normalize(zero))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("zero-norm");
  }

  @Test
  @DisplayName("CheckFinite: NaN and Infinity must throw IllegalArgumentException")
  void testNonFiniteValuesThrowException() {
    float[] withNan = {1.0f, Float.NaN, 3.0f};
    float[] withInf = {1.0f, Float.POSITIVE_INFINITY, 3.0f};

    assertThatThrownBy(() -> VectorUtils.checkFinite(withNan))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("non-finite");

    assertThatThrownBy(() -> VectorUtils.checkFinite(withInf))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("non-finite");
  }

  @Test
  @DisplayName("CheckDimension: Mismatched dimension must throw IllegalArgumentException")
  void testCheckDimension() {
    float[] v = {1.0f, 2.0f};

    assertThatThrownBy(() -> VectorUtils.checkDimension(v, 3))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("expected 3, but got 2");
  }
}
