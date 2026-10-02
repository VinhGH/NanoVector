package com.nanovector.server.controller;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.nanovector.server.NanoVectorServerApplication;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

@SpringBootTest(classes = NanoVectorServerApplication.class)
@AutoConfigureMockMvc
@DisplayName("OpenAPI, Swagger UI, and Actuator Health Integration Tests")
class ServerDocumentationAndActuatorTest {

  static {
    System.setProperty("spring.classformat.ignore", "true");
  }

  @Autowired private MockMvc mockMvc;

  @Test
  @DisplayName("OpenAPI JSON schema endpoint (/v3/api-docs) returns comprehensive API contract")
  void testOpenApiJsonDocs() throws Exception {
    MvcResult result =
        mockMvc
            .perform(get("/v3/api-docs"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.openapi").isNotEmpty())
            .andExpect(jsonPath("$.info.title").value("NanoVector Server REST API"))
            .andExpect(jsonPath("$.paths['/api/v1/indexes']").exists())
            .andExpect(jsonPath("$.paths['/api/v1/indexes/{name}/query']").exists())
            .andExpect(jsonPath("$.paths['/api/v1/indexes/{name}/vectors']").exists())
            .andExpect(jsonPath("$.paths['/api/v1/indexes/{name}/save']").exists())
            .andExpect(jsonPath("$.paths['/api/v1/indexes/load']").exists())
            .andExpect(jsonPath("$.components.schemas.CreateIndexRequest").exists())
            .andExpect(jsonPath("$.components.schemas.QueryRequest").exists())
            .andExpect(jsonPath("$.components.schemas.ErrorResponse").exists())
            .andReturn();

    String content = result.getResponse().getContentAsString();
    assertThat(content).contains("FLAT").contains("HNSW").contains("EUCLIDEAN");
  }

  @Test
  @DisplayName("Swagger UI is accessible and returns HTML dashboard")
  void testSwaggerUiDashboard() throws Exception {
    mockMvc.perform(get("/swagger-ui/index.html")).andExpect(status().isOk());
  }

  @Test
  @DisplayName(
      "Actuator health check (/actuator/health) returns READY status without leaking sensitive details")
  void testActuatorHealthCheck() throws Exception {
    mockMvc
        .perform(get("/actuator/health"))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.status").value("UP"))
        .andExpect(jsonPath("$.components.nanoVector.status").value("UP"))
        .andExpect(jsonPath("$.components.nanoVector.details.service").value("NanoVector Server"))
        .andExpect(jsonPath("$.components.nanoVector.details.status").value("READY"))
        .andExpect(jsonPath("$.components.nanoVector.details.activeIndexes").isNumber())
        // Invariant: No sensitive absolute paths leaked
        .andExpect(jsonPath("$.components.nanoVector.details.baseDir").doesNotExist())
        .andExpect(jsonPath("$.components.nanoVector.details.path").doesNotExist());
  }
}
