package com.nanovector.server.dto;

/** Request model for loading a persisted index (.nvec) from the server data directory. */
public record LoadIndexRequest(String name, String fileName) {}
