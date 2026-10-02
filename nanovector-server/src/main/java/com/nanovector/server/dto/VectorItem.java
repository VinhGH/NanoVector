package com.nanovector.server.dto;

/** Individual vector item container inside an insert request. */
public record VectorItem(Long id, float[] values) {}
