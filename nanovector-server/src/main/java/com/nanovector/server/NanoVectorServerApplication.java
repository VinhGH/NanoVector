package com.nanovector.server;

import com.nanovector.server.configuration.NanoVectorServerConfiguration;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Import;

/**
 * NanoVector Server Entry Point.
 *
 * <p>Spring Boot 3 application hosting the NanoVector RESTful vector search engine on Java 25 LTS.
 */
@SpringBootApplication
@Import(NanoVectorServerConfiguration.class)
public class NanoVectorServerApplication {

  static {
    System.setProperty("spring.classformat.ignore", "true");
  }

  public static void main(String[] args) {
    SpringApplication.run(NanoVectorServerApplication.class, args);
  }
}
