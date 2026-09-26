package com.nanovector.core.storage;

/**
 * Read-only contract interface providing direct sequential access to vector data and identifiers
 * stored in {@link VectorStorage}.
 *
 * <p><b>Important performance and safety contract:</b> The primitive buffers returned by {@link
 * #vectorBuffer()} and {@link #externalIdBuffer()} are the internal engine buffers to avoid copying
 * memory during persistence and streaming. They must be treated strictly as <b>read-only by
 * contract</b> by all callers. Modifying the contents of these arrays will corrupt internal engine
 * invariants.
 */
public interface VectorDataView {

  /** Returns the total number of active vectors currently stored. */
  int size();

  /** Returns the dimensionality of each vector. */
  int dimension();

  /**
   * Returns the direct internal contiguous primitive float buffer.
   *
   * <p>Vectors are stored in row-major layout where vector {@code i} occupies the index range
   * {@code [i * dimension, (i + 1) * dimension - 1]}.
   *
   * <p><b>Read-only contract:</b> Callers must treat the returned array as read-only.
   */
  float[] vectorBuffer();

  /** Returns the external ID corresponding to the given internal index. */
  long getExternalId(int internalId);

  /**
   * Returns the direct internal array of external identifiers indexed by internal ID.
   *
   * <p>Only elements {@code 0} through {@code size() - 1} are valid.
   *
   * <p><b>Read-only contract:</b> Callers must treat the returned array as read-only.
   */
  long[] externalIdBuffer();
}
