package com.nanovector.server.dto;

/** Request model for k-NN similarity search over a vector index. */
public record QueryRequest(float[] vector, Integer k, Integer efSearch) {}
