package com.nanovector.persistence.format;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.nanovector.core.distance.DistanceMetric;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class NvecConstantsTest {

  @Test
  @DisplayName("Constants have expected values per FORMAT_SPEC_V1")
  void testConstantsValues() {
    assertThat(NvecConstants.MAGIC_BYTES).containsExactly('N', 'V', 'E', 'C');
    assertThat(NvecConstants.FORMAT_VERSION_1).isEqualTo((short) 1);
    assertThat(NvecConstants.ENDIANNESS_LITTLE).isEqualTo((byte) 1);
    assertThat(NvecConstants.HEADER_SIZE_BYTES).isEqualTo(32);
    assertThat(NvecConstants.RESERVED_BYTES_COUNT).isEqualTo(3);
    assertThat(NvecConstants.HEADER_PADDING_BYTES_COUNT).isEqualTo(12);
    assertThat(NvecConstants.METADATA_LENGTH_FIELD_BYTES).isEqualTo(4);
    assertThat(NvecConstants.FLAT_METADATA_PAYLOAD_BYTES).isEqualTo(0);
    assertThat(NvecConstants.HNSW_METADATA_PAYLOAD_BYTES).isEqualTo(24);
    assertThat(NvecConstants.FOOTER_CHECKSUM_SIZE_BYTES).isEqualTo(4);
    assertThat(NvecConstants.MIN_FILE_SIZE_BYTES).isEqualTo(40);
    assertThat(NvecConstants.FILE_EXTENSION).isEqualTo(".nvec");
  }

  @ParameterizedTest
  @EnumSource(DistanceMetric.class)
  @DisplayName("Metric code conversion round-trips for all metrics")
  void testMetricConversionRoundTrip(DistanceMetric metric) {
    byte code = NvecConstants.metricToCode(metric);
    DistanceMetric resolved = NvecConstants.codeToMetric(code);
    assertThat(resolved).isEqualTo(metric);
  }

  @Test
  @DisplayName("Metric code mapping matches explicit specification constants")
  void testExplicitMetricCodes() {
    assertThat(NvecConstants.metricToCode(DistanceMetric.EUCLIDEAN))
        .isEqualTo(NvecConstants.METRIC_EUCLIDEAN);
    assertThat(NvecConstants.metricToCode(DistanceMetric.COSINE))
        .isEqualTo(NvecConstants.METRIC_COSINE);
    assertThat(NvecConstants.metricToCode(DistanceMetric.DOT_PRODUCT))
        .isEqualTo(NvecConstants.METRIC_DOT_PRODUCT);

    assertThat(NvecConstants.codeToMetric(NvecConstants.METRIC_EUCLIDEAN))
        .isEqualTo(DistanceMetric.EUCLIDEAN);
    assertThat(NvecConstants.codeToMetric(NvecConstants.METRIC_COSINE))
        .isEqualTo(DistanceMetric.COSINE);
    assertThat(NvecConstants.codeToMetric(NvecConstants.METRIC_DOT_PRODUCT))
        .isEqualTo(DistanceMetric.DOT_PRODUCT);
  }

  @Test
  @DisplayName("Invalid metric code throws IllegalArgumentException")
  void testInvalidMetricCode() {
    assertThatThrownBy(() -> NvecConstants.codeToMetric((byte) 0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> NvecConstants.codeToMetric((byte) 4))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Null metric throws NullPointerException")
  void testNullMetric() {
    assertThatThrownBy(() -> NvecConstants.metricToCode(null))
        .isInstanceOf(NullPointerException.class);
  }
}
