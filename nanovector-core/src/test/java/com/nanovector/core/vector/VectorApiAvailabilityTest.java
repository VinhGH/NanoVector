package com.nanovector.core.vector;

import static org.assertj.core.api.Assertions.assertThat;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class VectorApiAvailabilityTest {

  @Test
  @DisplayName("Java Vector API (jdk.incubator.vector) must be accessible and configured")
  void testVectorApiAvailability() {
    VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
    assertThat(species)
        .as("FloatVector.SPECIES_PREFERRED must be available via jdk.incubator.vector")
        .isNotNull();
    assertThat(species.length()).as("Species length must be positive").isGreaterThan(0);

    System.out.printf(
        "[Vector API] Runtime Preferred Species: %s (Shape: %s, Lanes: %d floats)%n",
        species, species.vectorShape(), species.length());
  }
}
