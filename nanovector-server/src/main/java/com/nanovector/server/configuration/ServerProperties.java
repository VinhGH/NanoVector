package com.nanovector.server.configuration;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/** Configuration properties for NanoVector Server. */
@Component
@ConfigurationProperties(prefix = "nanovector.server")
public class ServerProperties {

  /** Base storage directory for persisted index (.nvec) files. */
  private String dataDir = "data/indexes";

  /** Maximum allowed vector count per batch insert request. */
  private int maxBatchSize = 10_000;

  public String getDataDir() {
    return dataDir;
  }

  public void setDataDir(String dataDir) {
    this.dataDir = dataDir;
  }

  public int getMaxBatchSize() {
    return maxBatchSize;
  }

  public void setMaxBatchSize(int maxBatchSize) {
    this.maxBatchSize = maxBatchSize;
  }
}
