package com.nanovector.server.configuration;

import com.nanovector.core.search.SearchContextProvider;
import com.nanovector.server.concurrency.ConcurrentSearchSession;
import com.nanovector.server.registry.IndexRegistry;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** Spring Bean configuration declaring system-level components. */
@Configuration
public class NanoVectorServerConfiguration {

  @Bean
  public IndexRegistry indexRegistry() {
    return new IndexRegistry();
  }

  @Bean
  public SearchContextProvider searchContextProvider() {
    return SearchContextProvider.defaultProvider();
  }

  @Bean
  public ConcurrentSearchSession concurrentSearchSession(
      SearchContextProvider searchContextProvider) {
    return new ConcurrentSearchSession(searchContextProvider);
  }
}
