package com.nanovector.core.distance;

import java.util.Objects;

/**
 * Computes inverted Dot Product distance:
 * <pre>
 *   distance(a, b) = -dot(a, b)
 * </pre>
 * Negates the dot product to conform to the <b>Distance Minimization</b> contract,
 * where larger inner products map to smaller distance values.
 */
public final class ScalarDotProductDistance implements DistanceCalculator {

    @Override
    public float distance(float[] a, float[] b) {
        Objects.requireNonNull(a, "Vector 'a' must not be null");
        Objects.requireNonNull(b, "Vector 'b' must not be null");
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Vector dimension mismatch: a.length=" + a.length + ", b.length=" + b.length);
        }

        float dot = 0.0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return -dot;
    }

    @Override
    public float distance(float[] buffer, int offset, float[] query) {
        Objects.requireNonNull(buffer, "Buffer must not be null");
        Objects.requireNonNull(query, "Query vector must not be null");
        if (offset < 0 || offset + query.length > buffer.length) {
            throw new IndexOutOfBoundsException(
                    "Offset and query length exceed buffer bounds: offset=" + offset
                            + ", query.length=" + query.length + ", buffer.length=" + buffer.length);
        }

        float dot = 0.0f;
        for (int i = 0; i < query.length; i++) {
            dot += buffer[offset + i] * query[i];
        }
        return -dot;
    }

    @Override
    public DistanceMetric metric() {
        return DistanceMetric.DOT_PRODUCT;
    }
}
