package com.nanovector.server.dto;

import java.util.List;

/** Response model returning k-NN similarity search results and execution metrics. */
public record QueryResponse(
    String indexName, int k, List<SearchResultItem> results, long tookMicros) {}
