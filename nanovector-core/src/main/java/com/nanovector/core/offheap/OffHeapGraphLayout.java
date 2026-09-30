package com.nanovector.core.offheap;

import com.nanovector.core.hnsw.HnswConfig;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Low-level off-heap memory layout for HNSW multi-layer graph topology.
 *
 * <p>Separates graph storage into two distinct native memory zones:
 *
 * <ul>
 *   <li><b>Layer 0 Segment:</b> Dedicated contiguous fixed-stride segment storing {@code degree}
 *       and neighbor slots bounded to {@code M0 + 8}. Because 100% of nodes participate in Layer 0
 *       and receive the majority of distance queries, addressing is purely arithmetic ($O(1)$)
 *       without pointer chasing:
 *       <pre>
 *         offset = internalId * layer0StrideBytes
 *       </pre>
 *   <li><b>Tiered Upper Layers Segment:</b> Flat sequential segment storing layers $l \ge 1$ for
 *       the fraction ($\approx 6\%$) of nodes with {@code maxLevel > 0}. Per-node upper layer
 *       offsets and {@code maxLevel} are recorded in a 16-byte aligned index table ({@code
 *       nodeMetaSegment}).
 * </ul>
 *
 * <p>All native memory is allocated through a Foreign Function &amp; Memory (FFM) {@link Arena},
 * ensuring deterministic deallocation on {@link #close()}.
 */
public final class OffHeapGraphLayout implements AutoCloseable {

  private static final int[] EMPTY_INT_ARRAY = new int[0];
  private static final int DEFAULT_INITIAL_CAPACITY = 1024;
  private static final int UPPER_SLOT_PADDING = 8;

  private final HnswConfig config;
  private final Arena arena;
  private final boolean ownsArena;

  private final int l0MaxSlots;
  private final long l0StrideBytes;
  private final int upperMaxSlots;
  private final long upperLayerStrideBytes;
  private final long nodeMetaStrideBytes;

  private int capacity;
  private int nodeCount;

  private MemorySegment layer0Segment;
  private MemorySegment nodeMetaSegment;
  private MemorySegment upperLayersSegment;
  private long upperLayersAllocatedBytes;
  private boolean isClosed;

  public OffHeapGraphLayout(HnswConfig config) {
    this(config, DEFAULT_INITIAL_CAPACITY);
  }

  public OffHeapGraphLayout(HnswConfig config, int initialCapacity) {
    this(config, initialCapacity, Arena.ofConfined(), true);
  }

  public OffHeapGraphLayout(HnswConfig config, int initialCapacity, Arena arena) {
    this(config, initialCapacity, arena, false);
  }

  private OffHeapGraphLayout(
      HnswConfig config, int initialCapacity, Arena arena, boolean ownsArena) {
    if (initialCapacity <= 0) {
      throw new IllegalArgumentException("Initial capacity must be positive: " + initialCapacity);
    }
    this.config = Objects.requireNonNull(config, "HnswConfig must not be null");
    this.arena = Objects.requireNonNull(arena, "Arena must not be null");
    this.ownsArena = ownsArena;

    this.capacity = initialCapacity;
    this.nodeCount = 0;

    this.l0MaxSlots = config.m0() + UPPER_SLOT_PADDING;
    this.l0StrideBytes = (1L + l0MaxSlots) * Integer.BYTES;

    this.upperMaxSlots = config.m() + UPPER_SLOT_PADDING;
    this.upperLayerStrideBytes = (1L + upperMaxSlots) * Integer.BYTES;
    this.nodeMetaStrideBytes = 16L; // 4B maxLevel, 4B pad, 8B upperOffset

    this.layer0Segment = arena.allocate((long) capacity * l0StrideBytes, 8);
    this.nodeMetaSegment = arena.allocate((long) capacity * nodeMetaStrideBytes, 8);

    long initialUpperBytes =
        Math.max(
            (long) (capacity / 8 + 16) * upperLayerStrideBytes * 4L, upperLayerStrideBytes * 16L);
    this.upperLayersSegment = arena.allocate(initialUpperBytes, 8);
    this.upperLayersAllocatedBytes = 0L;
    this.isClosed = false;
  }

  /**
   * Registers a new node into the graph layout.
   *
   * @param internalId internal index of the node (must match {@link #nodeCount()})
   * @param maxLevel the highest level this node participates in (0-based)
   */
  public void addNode(int internalId, int maxLevel) {
    checkClosed();
    if (internalId != nodeCount) {
      throw new IllegalArgumentException(
          "Node internalId must be " + nodeCount + ", but got: " + internalId);
    }
    if (maxLevel < 0) {
      throw new IllegalArgumentException("Max level must be non-negative: " + maxLevel);
    }

    ensureCapacity(nodeCount + 1);

    long metaOffset = (long) internalId * nodeMetaStrideBytes;
    nodeMetaSegment.set(ValueLayout.JAVA_INT, metaOffset, maxLevel);

    if (maxLevel == 0) {
      nodeMetaSegment.set(ValueLayout.JAVA_LONG, metaOffset + 8L, -1L);
    } else {
      long bytesNeeded = (long) maxLevel * upperLayerStrideBytes;
      ensureUpperCapacity(upperLayersAllocatedBytes + bytesNeeded);

      long nodeUpperOffset = upperLayersAllocatedBytes;
      nodeMetaSegment.set(ValueLayout.JAVA_LONG, metaOffset + 8L, nodeUpperOffset);

      // Initialize upper layer degrees to 0
      for (int l = 1; l <= maxLevel; l++) {
        long layerOffset = nodeUpperOffset + (long) (l - 1) * upperLayerStrideBytes;
        upperLayersSegment.set(ValueLayout.JAVA_INT, layerOffset, 0);
      }
      upperLayersAllocatedBytes += bytesNeeded;
    }

    // Initialize Layer 0 degree to 0
    long l0Offset = (long) internalId * l0StrideBytes;
    layer0Segment.set(ValueLayout.JAVA_INT, l0Offset, 0);

    nodeCount++;
  }

  /** Returns the max level for the given node. */
  public int maxLevel(int internalId) {
    checkClosed();
    checkNodeBounds(internalId);
    return nodeMetaSegment.get(ValueLayout.JAVA_INT, (long) internalId * nodeMetaStrideBytes);
  }

  /** Returns the degree of the given node at the specified layer. */
  public int degree(int internalId, int layer) {
    checkClosed();
    checkLayerBounds(internalId, layer);
    if (layer == 0) {
      return layer0Segment.get(ValueLayout.JAVA_INT, (long) internalId * l0StrideBytes);
    }
    long nodeUpperOffset =
        nodeMetaSegment.get(ValueLayout.JAVA_LONG, (long) internalId * nodeMetaStrideBytes + 8L);
    long layerOffset = nodeUpperOffset + (long) (layer - 1) * upperLayerStrideBytes;
    return upperLayersSegment.get(ValueLayout.JAVA_INT, layerOffset);
  }

  /** Returns the neighbor internal ID at index {@code index} for the given node and layer. */
  public int getNeighbor(int internalId, int layer, int index) {
    checkClosed();
    checkLayerBounds(internalId, layer);
    int deg = degree(internalId, layer);
    if (index < 0 || index >= deg) {
      throw new IndexOutOfBoundsException(
          "Neighbor index " + index + " out of bounds, degree is " + deg);
    }
    if (layer == 0) {
      long offset = (long) internalId * l0StrideBytes + (1L + index) * Integer.BYTES;
      return layer0Segment.get(ValueLayout.JAVA_INT, offset);
    }
    long nodeUpperOffset =
        nodeMetaSegment.get(ValueLayout.JAVA_LONG, (long) internalId * nodeMetaStrideBytes + 8L);
    long offset =
        nodeUpperOffset + (long) (layer - 1) * upperLayerStrideBytes + (1L + index) * Integer.BYTES;
    return upperLayersSegment.get(ValueLayout.JAVA_INT, offset);
  }

  /** Returns an array containing all neighbor internal IDs for the given node at that layer. */
  public int[] getNeighbors(int internalId, int layer) {
    checkClosed();
    checkLayerBounds(internalId, layer);
    int deg = degree(internalId, layer);
    if (deg == 0) {
      return EMPTY_INT_ARRAY;
    }
    int[] result = new int[deg];
    if (layer == 0) {
      long offset = (long) internalId * l0StrideBytes + Integer.BYTES;
      MemorySegment.copy(layer0Segment, ValueLayout.JAVA_INT, offset, result, 0, deg);
    } else {
      long nodeUpperOffset =
          nodeMetaSegment.get(ValueLayout.JAVA_LONG, (long) internalId * nodeMetaStrideBytes + 8L);
      long offset = nodeUpperOffset + (long) (layer - 1) * upperLayerStrideBytes + Integer.BYTES;
      MemorySegment.copy(upperLayersSegment, ValueLayout.JAVA_INT, offset, result, 0, deg);
    }
    return result;
  }

  /** Overwrites the entire neighbor list of a node at the specified layer. */
  public void setNeighbors(int internalId, int layer, int[] newNeighbors) {
    checkClosed();
    checkLayerBounds(internalId, layer);
    Objects.requireNonNull(newNeighbors, "newNeighbors must not be null");

    if (layer == 0) {
      if (newNeighbors.length > l0MaxSlots) {
        throw new IllegalArgumentException(
            "Neighbor count " + newNeighbors.length + " exceeds Layer 0 slot limit " + l0MaxSlots);
      }
      long baseOffset = (long) internalId * l0StrideBytes;
      layer0Segment.set(ValueLayout.JAVA_INT, baseOffset, newNeighbors.length);
      if (newNeighbors.length > 0) {
        MemorySegment.copy(
            newNeighbors,
            0,
            layer0Segment,
            ValueLayout.JAVA_INT,
            baseOffset + Integer.BYTES,
            newNeighbors.length);
      }
    } else {
      if (newNeighbors.length > upperMaxSlots) {
        throw new IllegalArgumentException(
            "Neighbor count "
                + newNeighbors.length
                + " exceeds upper layer slot limit "
                + upperMaxSlots);
      }
      long nodeUpperOffset =
          nodeMetaSegment.get(ValueLayout.JAVA_LONG, (long) internalId * nodeMetaStrideBytes + 8L);
      long baseOffset = nodeUpperOffset + (long) (layer - 1) * upperLayerStrideBytes;
      upperLayersSegment.set(ValueLayout.JAVA_INT, baseOffset, newNeighbors.length);
      if (newNeighbors.length > 0) {
        MemorySegment.copy(
            newNeighbors,
            0,
            upperLayersSegment,
            ValueLayout.JAVA_INT,
            baseOffset + Integer.BYTES,
            newNeighbors.length);
      }
    }
  }

  /** Adds a single neighbor to the node's adjacency list at the specified layer. */
  public void addNeighbor(int internalId, int layer, int neighborId) {
    checkClosed();
    checkLayerBounds(internalId, layer);

    if (layer == 0) {
      long baseOffset = (long) internalId * l0StrideBytes;
      int deg = layer0Segment.get(ValueLayout.JAVA_INT, baseOffset);
      if (deg >= l0MaxSlots) {
        throw new IllegalStateException(
            "Layer 0 max degree slots (" + l0MaxSlots + ") exceeded for node " + internalId);
      }
      layer0Segment.set(ValueLayout.JAVA_INT, baseOffset + (1L + deg) * Integer.BYTES, neighborId);
      layer0Segment.set(ValueLayout.JAVA_INT, baseOffset, deg + 1);
    } else {
      long nodeUpperOffset =
          nodeMetaSegment.get(ValueLayout.JAVA_LONG, (long) internalId * nodeMetaStrideBytes + 8L);
      long baseOffset = nodeUpperOffset + (long) (layer - 1) * upperLayerStrideBytes;
      int deg = upperLayersSegment.get(ValueLayout.JAVA_INT, baseOffset);
      if (deg >= upperMaxSlots) {
        throw new IllegalStateException(
            "Upper layer max degree slots (" + upperMaxSlots + ") exceeded for node " + internalId);
      }
      upperLayersSegment.set(
          ValueLayout.JAVA_INT, baseOffset + (1L + deg) * Integer.BYTES, neighborId);
      upperLayersSegment.set(ValueLayout.JAVA_INT, baseOffset, deg + 1);
    }
  }

  /** Removes a neighbor from the node's adjacency list at the specified layer if present. */
  public void removeNeighbor(int internalId, int layer, int neighborId) {
    checkClosed();
    checkLayerBounds(internalId, layer);

    if (layer == 0) {
      long baseOffset = (long) internalId * l0StrideBytes;
      int deg = layer0Segment.get(ValueLayout.JAVA_INT, baseOffset);
      int idx = -1;
      for (int i = 0; i < deg; i++) {
        if (layer0Segment.get(ValueLayout.JAVA_INT, baseOffset + (1L + i) * Integer.BYTES)
            == neighborId) {
          idx = i;
          break;
        }
      }
      if (idx == -1) {
        return;
      }
      int shiftCount = deg - idx - 1;
      if (shiftCount > 0) {
        MemorySegment.copy(
            layer0Segment,
            ValueLayout.JAVA_INT,
            baseOffset + (1L + idx + 1) * Integer.BYTES,
            layer0Segment,
            ValueLayout.JAVA_INT,
            baseOffset + (1L + idx) * Integer.BYTES,
            shiftCount);
      }
      layer0Segment.set(ValueLayout.JAVA_INT, baseOffset, deg - 1);
    } else {
      long nodeUpperOffset =
          nodeMetaSegment.get(ValueLayout.JAVA_LONG, (long) internalId * nodeMetaStrideBytes + 8L);
      long baseOffset = nodeUpperOffset + (long) (layer - 1) * upperLayerStrideBytes;
      int deg = upperLayersSegment.get(ValueLayout.JAVA_INT, baseOffset);
      int idx = -1;
      for (int i = 0; i < deg; i++) {
        if (upperLayersSegment.get(ValueLayout.JAVA_INT, baseOffset + (1L + i) * Integer.BYTES)
            == neighborId) {
          idx = i;
          break;
        }
      }
      if (idx == -1) {
        return;
      }
      int shiftCount = deg - idx - 1;
      if (shiftCount > 0) {
        MemorySegment.copy(
            upperLayersSegment,
            ValueLayout.JAVA_INT,
            baseOffset + (1L + idx + 1) * Integer.BYTES,
            upperLayersSegment,
            ValueLayout.JAVA_INT,
            baseOffset + (1L + idx) * Integer.BYTES,
            shiftCount);
      }
      upperLayersSegment.set(ValueLayout.JAVA_INT, baseOffset, deg - 1);
    }
  }

  /** Checks whether the node contains the specified neighbor at that layer. */
  public boolean hasNeighbor(int internalId, int layer, int neighborId) {
    checkClosed();
    checkLayerBounds(internalId, layer);
    if (layer == 0) {
      long baseOffset = (long) internalId * l0StrideBytes;
      int deg = layer0Segment.get(ValueLayout.JAVA_INT, baseOffset);
      for (int i = 0; i < deg; i++) {
        if (layer0Segment.get(ValueLayout.JAVA_INT, baseOffset + (1L + i) * Integer.BYTES)
            == neighborId) {
          return true;
        }
      }
      return false;
    } else {
      long nodeUpperOffset =
          nodeMetaSegment.get(ValueLayout.JAVA_LONG, (long) internalId * nodeMetaStrideBytes + 8L);
      long baseOffset = nodeUpperOffset + (long) (layer - 1) * upperLayerStrideBytes;
      int deg = upperLayersSegment.get(ValueLayout.JAVA_INT, baseOffset);
      for (int i = 0; i < deg; i++) {
        if (upperLayersSegment.get(ValueLayout.JAVA_INT, baseOffset + (1L + i) * Integer.BYTES)
            == neighborId) {
          return true;
        }
      }
      return false;
    }
  }

  private void ensureCapacity(int requiredCapacity) {
    if (requiredCapacity <= capacity) {
      return;
    }
    int newCap = Math.max(capacity * 2, requiredCapacity);

    MemorySegment newL0 = arena.allocate((long) newCap * l0StrideBytes, 8);
    MemorySegment.copy(layer0Segment, 0L, newL0, 0L, (long) nodeCount * l0StrideBytes);
    this.layer0Segment = newL0;

    MemorySegment newMeta = arena.allocate((long) newCap * nodeMetaStrideBytes, 8);
    MemorySegment.copy(nodeMetaSegment, 0L, newMeta, 0L, (long) nodeCount * nodeMetaStrideBytes);
    this.nodeMetaSegment = newMeta;

    this.capacity = newCap;
  }

  private void ensureUpperCapacity(long requiredBytes) {
    if (requiredBytes <= upperLayersSegment.byteSize()) {
      return;
    }
    long newSize = Math.max(upperLayersSegment.byteSize() * 2L, requiredBytes + 65536L);
    MemorySegment newUpper = arena.allocate(newSize, 8);
    MemorySegment.copy(upperLayersSegment, 0L, newUpper, 0L, upperLayersAllocatedBytes);
    this.upperLayersSegment = newUpper;
  }

  private void checkNodeBounds(int internalId) {
    if (internalId < 0 || internalId >= nodeCount) {
      throw new IndexOutOfBoundsException(
          "Node internalId " + internalId + " out of bounds, node count: " + nodeCount);
    }
  }

  private void checkLayerBounds(int internalId, int layer) {
    checkNodeBounds(internalId);
    int maxL = maxLevel(internalId);
    if (layer < 0 || layer > maxL) {
      throw new IndexOutOfBoundsException(
          "Layer " + layer + " out of bounds for node " + internalId + " (maxLevel=" + maxL + ")");
    }
  }

  private void checkClosed() {
    if (isClosed) {
      throw new IllegalStateException("OffHeapGraphLayout has been closed");
    }
  }

  /** Total number of nodes registered in the layout. */
  public int nodeCount() {
    return nodeCount;
  }

  /** Current allocated node capacity before segment resize. */
  public int capacity() {
    return capacity;
  }

  /** Total native memory allocated across all active segments (bytes). */
  public long nativeAllocatedBytes() {
    return layer0Segment.byteSize() + nodeMetaSegment.byteSize() + upperLayersSegment.byteSize();
  }

  public HnswConfig config() {
    return config;
  }

  public int l0MaxSlots() {
    return l0MaxSlots;
  }

  public int upperMaxSlots() {
    return upperMaxSlots;
  }

  public long l0StrideBytes() {
    return l0StrideBytes;
  }

  public long upperLayerStrideBytes() {
    return upperLayerStrideBytes;
  }

  @Override
  public void close() {
    if (!isClosed) {
      isClosed = true;
      if (ownsArena && arena.scope().isAlive()) {
        arena.close();
      }
    }
  }
}
