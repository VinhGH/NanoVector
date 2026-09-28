package com.nanovector.core.quantization;

import java.util.Objects;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-accelerated Asymmetric Euclidean Distance (ADC) using the Java Vector API ({@code
 * jdk.incubator.vector}).
 *
 * <p>Loads unsigned 8-bit quantized values directly into SIMD vector lanes, expands to integers and
 * floats, applies affine reconstruction ($v_{\min} + \text{scale} \cdot q$) via fused multiply-add
 * (FMA), and evaluates squared differences against full-precision FP32 query lanes in parallel.
 *
 * <p>Includes scalar tail handling for vector dimensions that are not exact multiples of SIMD lane
 * width.
 */
public final class VectorQuantizedEuclideanDistance implements QuantizedEuclideanDistance {

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

  private final ScalarQuantizedEuclideanDistance fallback = new ScalarQuantizedEuclideanDistance();

  @Override
  public float distance(byte[] buffer, int offset, float min, float scale, float[] query) {
    if (!SIMD_SUPPORTED) {
      return fallback.distance(buffer, offset, min, scale, query);
    }

    Objects.requireNonNull(buffer, "buffer must not be null");
    Objects.requireNonNull(query, "query must not be null");
    int length = query.length;
    if (offset < 0 || offset + length > buffer.length) {
      throw new IndexOutOfBoundsException(
          String.format(
              "Buffer slice out of bounds: offset=%d, length=%d, buffer.length=%d",
              offset, length, buffer.length));
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
      ByteVector bv = ByteVector.fromArray(BYTE_SPECIES, buffer, offset + i);
      IntVector iv = (IntVector) bv.convertShape(VectorOperators.B2I, INT_SPECIES, 0);
      FloatVector fv = (FloatVector) iv.and(0xFF).convert(VectorOperators.I2F, 0);
      FloatVector vRecon = fv.fma(vScale, vMin);
      FloatVector vQuery = FloatVector.fromArray(FLOAT_SPECIES, query, i);
      FloatVector diff = vQuery.sub(vRecon);
      sumVec = diff.fma(diff, sumVec);
    }

    float sum = sumVec.reduceLanes(VectorOperators.ADD);

    // Scalar tail loop for remaining dimensions
    for (; i < length; i++) {
      int q = buffer[offset + i] & 0xFF;
      float recon = min + scale * q;
      float diff = query[i] - recon;
      sum += diff * diff;
    }

    return sum;
  }
}
