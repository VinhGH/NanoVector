package com.nanovector.server.configuration;

import com.nanovector.core.search.SearchContextProvider;
import com.nanovector.server.concurrency.ConcurrentSearchSession;
import com.nanovector.server.controller.IndexController;
import com.nanovector.server.exception.GlobalExceptionHandler;
import com.nanovector.server.registry.IndexRegistry;
import com.nanovector.server.service.IndexService;
import com.nanovector.server.service.StoragePathResolver;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** Spring Bean configuration declaring system-level components. */
@Configuration
public class NanoVectorServerConfiguration {

  @Bean
  public ServerProperties serverProperties() {
    return new ServerProperties();
  }

  @Bean
  public StoragePathResolver storagePathResolver(ServerProperties serverProperties) {
    return new StoragePathResolver(serverProperties.getDataDir());
  }

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

  @Bean
  public IndexService indexService(
      IndexRegistry indexRegistry,
      ConcurrentSearchSession concurrentSearchSession,
      StoragePathResolver storagePathResolver,
      ServerProperties serverProperties) {
    return new IndexService(
        indexRegistry, concurrentSearchSession, storagePathResolver, serverProperties);
  }

  @Bean
  public IndexController indexController(IndexService indexService) {
    return new IndexController(indexService);
  }

  @Bean
  public GlobalExceptionHandler globalExceptionHandler() {
    return new GlobalExceptionHandler();
  }
}
