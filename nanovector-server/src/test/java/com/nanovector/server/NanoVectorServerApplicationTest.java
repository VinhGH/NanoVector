package com.nanovector.server;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.core.search.SearchContextProvider;
import com.nanovector.server.concurrency.ConcurrentSearchSession;
import com.nanovector.server.registry.IndexRegistry;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest(classes = NanoVectorServerApplication.class)
@DisplayName("NanoVectorServerApplication Context Loading Test")
class NanoVectorServerApplicationTest {

  static {
    System.setProperty("spring.classformat.ignore", "true");
  }

  @Autowired private IndexRegistry indexRegistry;

  @Autowired private SearchContextProvider searchContextProvider;

  @Autowired private ConcurrentSearchSession concurrentSearchSession;

  @Test
  @DisplayName("Spring application context loads and wires required foundation beans")
  void contextLoads() {
    assertThat(indexRegistry).isNotNull();
    assertThat(searchContextProvider).isNotNull();
    assertThat(concurrentSearchSession).isNotNull();
    assertThat(concurrentSearchSession.contextProvider()).isSameAs(searchContextProvider);
  }
}
