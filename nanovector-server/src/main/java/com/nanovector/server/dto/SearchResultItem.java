package com.nanovector.server.dto;

/** Individual k-NN search result item containing external vector ID and distance score. */
public record SearchResultItem(long id, float score) {}
