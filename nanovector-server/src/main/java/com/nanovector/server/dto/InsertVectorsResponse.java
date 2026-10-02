package com.nanovector.server.dto;

/** Response model returning count of inserted vectors and updated total size. */
public record InsertVectorsResponse(int insertedCount, int totalSize) {}
