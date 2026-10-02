package com.nanovector.server.dto;

/** Request model for persisting an in-memory index to disk in NVEC v1 format. */
public record SaveIndexRequest(String fileName) {}
