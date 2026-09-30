package com.nanovector.core.offheap;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-accelerated Asymmetric Euclidean Distance (ADC) evaluated directly from off-heap {@link
 * MemorySegment} buffers using the Java Vector API ({@code jdk.incubator.vector}).
 *
 * <p>Uses {@link ByteVector#fromMemorySegment} to load SIMD lanes directly from native memory,
 * expanding to integers and floats, applying FMA affine reconstruction, and computing squared
 * differences in parallel.
 */
public final class OffHeapVectorQuantizedEuclideanDistance
    implements OffHeapQuantizedEuclideanDistance {

  private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
  private static final VectorSpecies<Integer> INT_SPECIES;
  private static final VectorSpecies<Byte> BYTE_SPECIES;
  private static final boolean SIMD_SUPPORTED;

  static {
    boolean supported = false;
    VectorSpecies<Integer> intSp = null;
    VectorSpecies<Byte> byteSp = null;
    try {
      int byteBitSize = FLOAT_SPECIES.vectorBitSize() / 4;
      if (byteBitSize >= 64) {
        VectorShape byteShape = VectorShape.forBitSize(byteBitSize);
        intSp = FLOAT_SPECIES.withLanes(int.class);
        byteSp = FLOAT_SPECIES.withLanes(byte.class).withShape(byteShape);
        supported = (intSp != null && byteSp != null);
      }
    } catch (Throwable t) {
      supported = false;
    }
    SIMD_SUPPORTED = supported;
    INT_SPECIES = intSp;
    BYTE_SPECIES = byteSp;
  }

  private final OffHeapScalarQuantizedEuclideanDistance fallback =
      new OffHeapScalarQuantizedEuclideanDistance();

  @Override
  public float distance(MemorySegment segment, long offset, float min, float scale, float[] query) {
    if (!SIMD_SUPPORTED) {
      return fallback.distance(segment, offset, min, scale, query);
    }

    Objects.requireNonNull(segment, "segment must not be null");
    Objects.requireNonNull(query, "query must not be null");
    int length = query.length;
    if (offset < 0 || offset + length > segment.byteSize()) {
      throw new IndexOutOfBoundsException(
          String.format(
              "MemorySegment slice out of bounds: offset=%d, length=%d, segment.byteSize=%d",
              offset, length, segment.byteSize()));
    }

    if (scale == 0.0f) {
      int upperBound = FLOAT_SPECIES.loopBound(length);
      FloatVector vMin = FloatVector.broadcast(FLOAT_SPECIES, min);
      FloatVector sumVec = FloatVector.zero(FLOAT_SPECIES);
      int i = 0;
      for (; i < upperBound; i += FLOAT_SPECIES.length()) {
        FloatVector vQuery = FloatVector.fromArray(FLOAT_SPECIES, query, i);
        FloatVector diff = vQuery.sub(vMin);
        sumVec = diff.fma(diff, sumVec);
      }
      float sum = sumVec.reduceLanes(VectorOperators.ADD);
      for (; i < length; i++) {
        float diff = query[i] - min;
        sum += diff * diff;
      }
      return sum;
    }

    int upperBound = FLOAT_SPECIES.loopBound(length);
    FloatVector vMin = FloatVector.broadcast(FLOAT_SPECIES, min);
    FloatVector vScale = FloatVector.broadcast(FLOAT_SPECIES, scale);
    FloatVector sumVec = FloatVector.zero(FLOAT_SPECIES);

    int i = 0;
    for (; i < upperBound; i += FLOAT_SPECIES.length()) {
      ByteVector bv =
          ByteVector.fromMemorySegment(BYTE_SPECIES, segment, offset + i, ByteOrder.nativeOrder());
      IntVector iv = (IntVector) bv.convertShape(VectorOperators.B2I, INT_SPECIES, 0);
      FloatVector fv = (FloatVector) iv.and(0xFF).convert(VectorOperators.I2F, 0);
      FloatVector vRecon = fv.fma(vScale, vMin);
      FloatVector vQuery = FloatVector.fromArray(FLOAT_SPECIES, query, i);
      FloatVector diff = vQuery.sub(vRecon);
      sumVec = diff.fma(diff, sumVec);
    }

    float sum = sumVec.reduceLanes(VectorOperators.ADD);

    // Scalar tail loop
    for (; i < length; i++) {
      int q = segment.get(ValueLayout.JAVA_BYTE, offset + i) & 0xFF;
      float recon = min + scale * q;
      float diff = query[i] - recon;
      sum += diff * diff;
    }

    return sum;
  }
}
