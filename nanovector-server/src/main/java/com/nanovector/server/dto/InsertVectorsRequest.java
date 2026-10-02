package com.nanovector.server.dto;

import java.util.List;

/** Request model for inserting a batch of vectors into a managed index. */
public record InsertVectorsRequest(List<VectorItem> vectors) {}
