package com.nanovector.server.dto;

/** Response model returning persistence outcome and target file details. */
public record SaveIndexResponse(String indexName, String fileName, int size, long fileSizeBytes) {}
