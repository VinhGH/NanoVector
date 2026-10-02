package com.nanovector.server.controller;

import static org.assertj.core.api.Assertions.assertThat;

import com.nanovector.server.NanoVectorServerApplication;
import com.nanovector.server.dto.CreateIndexRequest;
import com.nanovector.server.dto.InsertVectorsRequest;
import com.nanovector.server.dto.QueryRequest;
import com.nanovector.server.dto.QueryResponse;
import com.nanovector.server.dto.VectorItem;
import com.nanovector.server.registry.IndexRegistry;
import com.nanovector.server.registry.ManagedIndex;
import com.nanovector.server.service.IndexService;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest(classes = NanoVectorServerApplication.class)
@DisplayName("Index REST Concurrency and Lifecycle Integration Tests")
class IndexConcurrencyIntegrationTest {

  static {
    System.setProperty("spring.classformat.ignore", "true");
  }

  @Autowired private IndexService indexService;

  @Autowired private IndexRegistry indexRegistry;

  @Test
  @DisplayName(
      "Concurrent query operations during batch inserts obey locking contract without data corruption")
  void testConcurrentQueryDuringInsertObeysLockingContract() throws Exception {
    String indexName = "concurrent_rw_idx";
    indexService.createIndex(new CreateIndexRequest(indexName, "HNSW", 4, "EUCLIDEAN", null));

    // Seed with 20 initial vectors
    List<VectorItem> initial = new ArrayList<>();
    for (int i = 0; i < 20; i++) {
      initial.add(new VectorItem((long) i, new float[] {i * 0.1f, 0.2f, 0.3f, 0.4f}));
    }
    indexService.insertVectors(indexName, new InsertVectorsRequest(initial));

    int numReaderThreads = 8;
    int numRounds = 30;
    ExecutorService readers = Executors.newFixedThreadPool(numReaderThreads);
    ExecutorService writer = Executors.newSingleThreadExecutor();

    AtomicBoolean stopFlag = new AtomicBoolean(false);
    AtomicInteger queryCount = new AtomicInteger(0);
    List<Future<?>> readerFutures = new ArrayList<>();

    // Launch readers
    for (int t = 0; t < numReaderThreads; t++) {
      readerFutures.add(
          readers.submit(
              () -> {
                while (!stopFlag.get()) {
                  QueryResponse resp =
                      indexService.query(
                          indexName, new QueryRequest(new float[] {0.5f, 0.2f, 0.3f, 0.4f}, 5, 32));
                  assertThat(resp.results()).isNotEmpty();
                  queryCount.incrementAndGet();
                  Thread.yield();
                }
              }));
    }

    // Launch writers concurrently inserting new items
    Future<?> writerFuture =
        writer.submit(
            () -> {
              for (int round = 0; round < numRounds; round++) {
                List<VectorItem> batch = new ArrayList<>();
                long baseId = 1000L + (round * 5);
                for (int j = 0; j < 5; j++) {
                  batch.add(
                      new VectorItem(
                          baseId + j, new float[] {(baseId + j) * 0.01f, 0.2f, 0.3f, 0.4f}));
                }
                indexService.insertVectors(indexName, new InsertVectorsRequest(batch));
                try {
                  Thread.sleep(5);
                } catch (InterruptedException e) {
                  Thread.currentThread().interrupt();
                  break;
                }
              }
            });

    writerFuture.get(10, TimeUnit.SECONDS);
    stopFlag.set(true);

    for (Future<?> f : readerFutures) {
      f.get(10, TimeUnit.SECONDS);
    }

    readers.shutdown();
    writer.shutdown();

    assertThat(queryCount.get()).isGreaterThan(50);
    assertThat(indexService.getIndex(indexName).size()).isEqualTo(20 + numRounds * 5);
  }

  @Test
  @DisplayName("Delete during active query drains in-flight read without causing use-after-free")
  void testDeleteDuringOngoingQueryDoesNotCauseUseAfterFree() throws Exception {
    String indexName = "drain_offheap_delete_idx";
    indexService.createIndex(
        new CreateIndexRequest(indexName, "HNSW_SQ8_OFFHEAP", 4, "EUCLIDEAN", null));

    List<VectorItem> items = new ArrayList<>();
    for (int i = 0; i < 50; i++) {
      items.add(new VectorItem((long) i, new float[] {i * 0.1f, 0.2f, 0.3f, 0.4f}));
    }
    indexService.insertVectors(indexName, new InsertVectorsRequest(items));

    ManagedIndex managedIndex = indexRegistry.getRequired(indexName);

    CountDownLatch readerInsideLock = new CountDownLatch(1);
    CountDownLatch readerCanFinish = new CountDownLatch(1);
    AtomicBoolean readerSuccess = new AtomicBoolean(false);

    // Reader thread acquires read lock and pauses inside
    Thread readerThread =
        new Thread(
            () -> {
              try {
                managedIndex.executeRead(
                    idx -> {
                      readerInsideLock.countDown();
                      try {
                        readerCanFinish.await(5, TimeUnit.SECONDS);
                      } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                      }
                      // Run query while inside lock
                      var res = idx.searchKnn(new float[] {0.1f, 0.2f, 0.3f, 0.4f}, 5);
                      assertThat(res).isNotEmpty();
                      readerSuccess.set(true);
                      return null;
                    });
              } catch (Exception e) {
                readerSuccess.set(false);
              }
            });
    readerThread.start();

    // Wait until reader is holding readLock
    assertThat(readerInsideLock.await(3, TimeUnit.SECONDS)).isTrue();

    // Start delete thread; it must block on writeLock until reader completes
    CountDownLatch deleteStarted = new CountDownLatch(1);
    AtomicBoolean deleteFinished = new AtomicBoolean(false);
    Thread deleteThread =
        new Thread(
            () -> {
              deleteStarted.countDown();
              indexService.deleteIndex(indexName);
              deleteFinished.set(true);
            });
    deleteThread.start();

    deleteStarted.await(2, TimeUnit.SECONDS);
    Thread.sleep(100); // Give delete thread time to attempt acquire writeLock

    // Verify delete is blocked waiting for reader
    assertThat(deleteFinished.get()).isFalse();

    // Allow reader to finish
    readerCanFinish.countDown();
    readerThread.join(3000);
    deleteThread.join(3000);

    assertThat(readerSuccess.get()).isTrue();
    assertThat(deleteFinished.get()).isTrue();
    assertThat(managedIndex.isClosed()).isTrue();
  }
}
