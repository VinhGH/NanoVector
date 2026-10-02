package com.nanovector.server.dto;

import java.util.Map;

/** Request model for creating a new vector index. */
public record CreateIndexRequest(
    String name, String type, Integer dimension, String metric, Map<String, Object> parameters) {}
