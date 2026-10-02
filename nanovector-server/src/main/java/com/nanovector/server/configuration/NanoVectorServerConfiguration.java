package com.nanovector.server.configuration;

import com.nanovector.core.search.SearchContextProvider;
import com.nanovector.server.concurrency.ConcurrentSearchSession;
import com.nanovector.server.controller.IndexController;
import com.nanovector.server.exception.GlobalExceptionHandler;
import com.nanovector.server.lifecycle.NanoVectorHealthIndicator;
import com.nanovector.server.registry.IndexRegistry;
import com.nanovector.server.service.IndexService;
import com.nanovector.server.service.StoragePathResolver;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
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

  @Bean
  public NanoVectorHealthIndicator nanoVectorHealthIndicator(IndexRegistry indexRegistry) {
    return new NanoVectorHealthIndicator(indexRegistry);
  }

  @Bean
  public OpenAPI nanoVectorOpenAPI() {
    return new OpenAPI()
        .info(
            new Info()
                .title("NanoVector Server REST API")
                .description(
                    "High-performance pure Java vector search engine hosting FLAT, HNSW, "
                        + "and Scalar Quantized (SQ8) indexes with hardware-accelerated SIMD kernels.")
                .version("0.1.0-SNAPSHOT")
                .contact(
                    new Contact()
                        .name("NanoVector Team")
                        .url("https://github.com/nanovector/nanovector"))
                .license(
                    new License()
                        .name("Apache 2.0")
                        .url("https://www.apache.org/licenses/LICENSE-2.0.html")));
  }
}
