package com.nanovector.core.offheap;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Loop-fused single-pass scalar calculation for Asymmetric Euclidean Distance (ADC) directly
 * against off-heap {@link MemorySegment} storage.
 *
 * <p>Zero Java heap allocations during distance computation.
 */
public final class OffHeapScalarQuantizedEuclideanDistance
    implements OffHeapQuantizedEuclideanDistance {

  @Override
  public float distance(MemorySegment segment, long offset, float min, float scale, float[] query) {
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
      float sum = 0.0f;
      for (int i = 0; i < length; i++) {
        float diff = query[i] - min;
        sum += diff * diff;
      }
      return sum;
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
      int q = segment.get(ValueLayout.JAVA_BYTE, offset + i) & 0xFF;
      float recon = min + scale * q;
      float diff = query[i] - recon;
      sum += diff * diff;
    }
    return sum;
  }
}
