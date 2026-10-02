# NanoVector Server Operations & REST API Guide

NanoVector Server hosts the high-performance in-memory vector search engine as a production-ready HTTP microservice built on Spring Boot 3.4.0 and Java 25 LTS.

---

## 🚀 Quickstart

### 1. Requirements
- **Java**: OpenJDK 25 LTS (`java -version` >= 25)
- **Maven**: Maven 3.9+ (or use the included `./mvnw.cmd` / `./mvnw` wrapper)
- **JVM Flags**: `--add-modules jdk.incubator.vector` (configured automatically via Surefire and Spring Boot plugins)

### 2. Building the Project
From the repository root:
```bash
# Verify all tests and compile all modules
./mvnw clean verify
```

### 3. Running NanoVector Server
You can start the server directly using Maven:
```bash
./mvnw spring-boot:run -pl nanovector-server
```
Or package and run the executable fat JAR:
```bash
./mvnw package -pl nanovector-server -am -DskipTests
java --add-modules jdk.incubator.vector -jar nanovector-server/target/nanovector-server-0.1.0-SNAPSHOT.jar
```

By default, the server binds to port `8080`.

---

## 🩺 Health Check & Monitoring

NanoVector Server integrates Spring Boot Actuator for operational health checks:

- **Endpoint**: `GET http://localhost:8080/actuator/health`
- **Response Format**:
```json
{
  "status": "UP",
  "components": {
    "nanoVector": {
      "status": "UP",
      "details": {
        "service": "NanoVector Server",
        "status": "READY",
        "activeIndexes": 2
      }
    }
  }
}
```

> [!NOTE]
> The health check strictly avoids exposing internal filesystem paths, memory addresses, or sensitive system details.

---

## 📖 Interactive Documentation (OpenAPI / Swagger UI)

NanoVector Server embeds interactive OpenAPI 3.0 documentation:

- **Swagger UI Dashboard**: [http://localhost:8080/swagger-ui.html](http://localhost:8080/swagger-ui.html)
- **Raw OpenAPI 3 JSON Schema**: [http://localhost:8080/v3/api-docs](http://localhost:8080/v3/api-docs)

---

## ⚙️ Configuration Properties

All settings can be customized in `application.properties` or overridden via command-line arguments and environment variables:

| Property Key | Default Value | Description |
| :--- | :--- | :--- |
| `server.port` | `8080` | HTTP port for REST and Actuator endpoints |
| `nanovector.server.data-dir` | `data/indexes` | Base storage directory for `.nvec` files |
| `nanovector.server.max-batch-size` | `10000` | Maximum vectors allowed in a single batch insert |
| `spring.classformat.ignore` | `true` | Permits execution on Java 25 runtime |

**Example command-line override:**
```bash
java --add-modules jdk.incubator.vector -jar nanovector-server.jar \
  --server.port=9090 \
  --nanovector.server.data-dir=/var/lib/nanovector/data \
  --nanovector.server.max-batch-size=50000
```

---

## 🔒 Security & Persistence Boundaries

1. **Path Traversal Protection (`StoragePathResolver`)**:
   - The server **never** accepts arbitrary absolute paths from client requests.
   - All files are strictly resolved inside `nanovector.server.data-dir`.
   - File names containing path separators (`/`, `\`, `..`) are rejected with `400 Bad Request`.
2. **Atomic Serialization**:
   - Save operations write first to a `.tmp` file and perform an atomic filesystem move.
3. **NVEC v1 Format Scope**:
   - Serialization to `.nvec` is currently supported for `FLAT` and `HNSW` (FP32) index topologies.
   - Attempting to serialize `HNSW_SQ8` or `HNSW_SQ8_OFFHEAP` returns `400 Bad Request` with an explanatory message.
4. **Lifecycle & Read-Write Locking Contract**:
   - **Queries** execute holding a shared read lock with isolated visited sets.
   - **Inserts** execute holding an exclusive write lock.
   - **Save** holds a shared read lock throughout the entire serialization duration, preventing torn reads.
   - **Delete** acquires an exclusive write lock, draining active queries before closing native memory segments (preventing use-after-free).

---

## 📡 REST API Reference & Examples

### 1. Create Index
- **Endpoint**: `POST /api/v1/indexes`
- **Status**: `201 Created`

```bash
curl -X POST http://localhost:8080/api/v1/indexes \
  -H "Content-Type: application/json" \
  -d '{
    "name": "products",
    "type": "HNSW",
    "dimension": 4,
    "metric": "EUCLIDEAN",
    "parameters": {
      "m": 16,
      "efConstruction": 200,
      "efSearch": 50
    }
  }'
```

Supported index types:
- `FLAT`: FP32 On-Heap exact scan
- `HNSW`: FP32 On-Heap approximate nearest-neighbor
- `HNSW_SQ8`: Quantized On-Heap 8-bit scalar quantization (EUCLIDEAN only)
- `HNSW_SQ8_OFFHEAP`: Quantized Native Off-Heap FFM (EUCLIDEAN only)

### 2. List All Indexes
- **Endpoint**: `GET /api/v1/indexes`
- **Status**: `200 OK`

```bash
curl -X GET http://localhost:8080/api/v1/indexes
```

### 3. Get Index Metadata
- **Endpoint**: `GET /api/v1/indexes/{name}`
- **Status**: `200 OK`

```bash
curl -X GET http://localhost:8080/api/v1/indexes/products
```

### 4. Insert Batch Vectors
- **Endpoint**: `POST /api/v1/indexes/{name}/vectors`
- **Status**: `200 OK`

```bash
curl -X POST http://localhost:8080/api/v1/indexes/products/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {
        "id": 1001,
        "values": [1.0, 0.0, 0.0, 0.0]
      },
      {
        "id": 1002,
        "values": [0.0, 1.0, 0.0, 0.0]
      }
    ]
  }'
```

### 5. Execute k-NN Query
- **Endpoint**: `POST /api/v1/indexes/{name}/query`
- **Status**: `200 OK`

```bash
curl -X POST http://localhost:8080/api/v1/indexes/products/query \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1.0, 0.1, 0.0, 0.0],
    "k": 5,
    "efSearch": 64
  }'
```

Response:
```json
{
  "indexName": "products",
  "k": 5,
  "results": [
    {
      "id": 1001,
      "score": 0.010000001
    },
    {
      "id": 1002,
      "score": 1.01
    }
  ],
  "tookMicros": 85
}
```

### 6. Persist Index to Disk (`.nvec`)
- **Endpoint**: `POST /api/v1/indexes/{name}/save`
- **Status**: `200 OK`

```bash
curl -X POST http://localhost:8080/api/v1/indexes/products/save \
  -H "Content-Type: application/json" \
  -d '{
    "fileName": "products.nvec"
  }'
```

### 7. Load Index from Disk
- **Endpoint**: `POST /api/v1/indexes/load`
- **Status**: `201 Created`

```bash
curl -X POST http://localhost:8080/api/v1/indexes/load \
  -H "Content-Type: application/json" \
  -d '{
    "name": "products_restored",
    "fileName": "products.nvec"
  }'
```

### 8. Delete Index
- **Endpoint**: `DELETE /api/v1/indexes/{name}`
- **Status**: `204 No Content`

```bash
curl -X DELETE http://localhost:8080/api/v1/indexes/products
```

---

## ⚠️ Standard Error Handling

When errors occur, endpoints return uniform JSON payloads:
```json
{
  "timestamp": "2026-10-02T10:15:30Z",
  "status": 400,
  "error": "Bad Request",
  "message": "Vector dimension mismatch for id 1002: expected 4, got 3",
  "path": "/api/v1/indexes/products/vectors"
}
```

| HTTP Status | Description |
| :--- | :--- |
| `400 Bad Request` | Validation invariant violation, malformed JSON, corrupt file, unsupported metric |
| `404 Not Found` | Index or file not found |
| `409 Conflict` | Duplicate index name or lifecycle conflict |
| `413 Payload Too Large` | Vector count exceeds `nanovector.server.max-batch-size` |
| `500 Internal Server Error` | Unhandled server error |
