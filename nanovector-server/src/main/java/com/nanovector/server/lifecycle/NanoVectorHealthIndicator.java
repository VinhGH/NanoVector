package com.nanovector.server.lifecycle;

import com.nanovector.server.registry.IndexRegistry;
import java.util.Objects;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.stereotype.Component;

/**
 * HealthIndicator reporting operational health of NanoVector Server and index registry status
 * without leaking internal filesystem details or sensitive system metrics.
 */
@Component
public class NanoVectorHealthIndicator implements HealthIndicator {

  private final IndexRegistry registry;

  public NanoVectorHealthIndicator(IndexRegistry registry) {
    this.registry = Objects.requireNonNull(registry, "registry must not be null");
  }

  @Override
  public Health health() {
    int activeCount = registry.size();
    return Health.up()
        .withDetail("service", "NanoVector Server")
        .withDetail("status", "READY")
        .withDetail("activeIndexes", activeCount)
        .build();
  }
}
