package com.nanovector.benchmark.quantization;

import com.nanovector.core.distance.DistanceMetric;
import com.nanovector.core.index.FlatIndex;
import com.nanovector.core.index.QuantizedFlatIndex;
import com.nanovector.core.model.SearchResult;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Recall and Accuracy Benchmark evaluating the quantization-induced recall trade-off of 8-bit
 * Scalar Quantization (SQ8) across dataset scales ($N \in \{1\text{K}, 10\text{K}, 50\text{K},
 * 100\text{K}\}$ at dimension $D=128$).
 *
 * <p>Experimental Controls:
 *
 * <ul>
 *   <li><b>Exact Ground Truth Oracle</b>: Full-precision FP32 {@link FlatIndex} (100% recall by
 *       definition).
 *   <li><b>Candidates Under Test</b>: {@link QuantizedFlatIndex} with Scalar ADC and SIMD ADC.
 *   <li><b>Isolation of Error</b>: Evaluated strictly via exhaustive $O(N)$ sequential scan to
 *       isolate coordinate quantization error from graph exploration / routing heuristics.
 *   <li><b>Workload</b>: $D=128$, Metric = Euclidean, $k=10$, 128 queries, seeds 42L (data) and
 *       12345L (queries).
 * </ul>
 */
public final class Sq8RecallAccuracyBenchmark {

  public static final int DEFAULT_DIMENSION = 128;
  public static final DistanceMetric DEFAULT_METRIC = DistanceMetric.EUCLIDEAN;
  public static final int DEFAULT_K = 10;
  public static final int DEFAULT_NUM_QUERIES = 128;
  public static final long DEFAULT_DATA_SEED = 42L;
  public static final long DEFAULT_QUERY_SEED = 12345L;
  public static final int[] DEFAULT_SCALES = {1_000, 10_000, 50_000, 100_000};

  public record RecallSweepRow(
      int n,
      int dimension,
      int k,
      int numQueries,
      double fp32Recall,
      double sq8ScalarRecall,
      double sq8SimdRecall,
      double recallLoss,
      double scalarSimdAgreement) {

    public String toMarkdownRow() {
      return String.format(
          "| %,10d | %16.2f%% | %17.2f%% | %15.2f%% | %11.2f%% | %21.2f%% |",
          n,
          fp32Recall * 100.0,
          sq8ScalarRecall * 100.0,
          sq8SimdRecall * 100.0,
          recallLoss * 100.0,
          scalarSimdAgreement * 100.0);
    }
  }

  public static RecallSweepRow evaluate(
      int n, int dimension, int k, int numQueries, long dataSeed, long querySeed) {
    Random dataRng = new Random(dataSeed);
    float[][] dataset = new float[n][dimension];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dimension; d++) {
        dataset[i][d] = dataRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    Random queryRng = new Random(querySeed);
    float[][] queries = new float[numQueries][dimension];
    for (int q = 0; q < numQueries; q++) {
      for (int d = 0; d < dimension; d++) {
        queries[q][d] = queryRng.nextFloat() * 2.0f - 1.0f;
      }
    }

    FlatIndex fp32Oracle = new FlatIndex(dimension, DEFAULT_METRIC, n, true);
    QuantizedFlatIndex sq8Scalar = new QuantizedFlatIndex(dimension, n, false);
    QuantizedFlatIndex sq8Simd = new QuantizedFlatIndex(dimension, n, true);

    for (int i = 0; i < n; i++) {
      fp32Oracle.insert(i, dataset[i]);
      sq8Scalar.insert(i, dataset[i]);
      sq8Simd.insert(i, dataset[i]);
    }

    double totalScalarRecall = 0.0;
    double totalSimdRecall = 0.0;
    double totalAgreement = 0.0;

    for (int q = 0; q < numQueries; q++) {
      float[] query = queries[q];
      List<SearchResult> gtResults = fp32Oracle.searchKnn(query, k);
      List<SearchResult> scalarResults = sq8Scalar.searchKnn(query, k);
      List<SearchResult> simdResults = sq8Simd.searchKnn(query, k);

      Set<Long> gtIds = new HashSet<>(k);
      for (SearchResult r : gtResults) {
        gtIds.add(r.id());
      }

      Set<Long> scalarIds = new HashSet<>(k);
      for (SearchResult r : scalarResults) {
        scalarIds.add(r.id());
      }

      int scalarMatches = 0;
      for (long id : scalarIds) {
        if (gtIds.contains(id)) {
          scalarMatches++;
        }
      }

      int simdMatches = 0;
      int mutualMatches = 0;
      for (SearchResult r : simdResults) {
        if (gtIds.contains(r.id())) {
          simdMatches++;
        }
        if (scalarIds.contains(r.id())) {
          mutualMatches++;
        }
      }

      totalScalarRecall += (double) scalarMatches / k;
      totalSimdRecall += (double) simdMatches / k;
      totalAgreement += (double) mutualMatches / k;
    }

    double meanScalarRecall = totalScalarRecall / numQueries;
    double meanSimdRecall = totalSimdRecall / numQueries;
    double meanAgreement = totalAgreement / numQueries;
    double recallLoss = 1.0 - meanSimdRecall;

    return new RecallSweepRow(
        n,
        dimension,
        k,
        numQueries,
        1.0,
        meanScalarRecall,
        meanSimdRecall,
        recallLoss,
        meanAgreement);
  }

  public static List<RecallSweepRow> runSweep() {
    List<RecallSweepRow> rows = new ArrayList<>();
    for (int n : DEFAULT_SCALES) {
      rows.add(
          evaluate(
              n,
              DEFAULT_DIMENSION,
              DEFAULT_K,
              DEFAULT_NUM_QUERIES,
              DEFAULT_DATA_SEED,
              DEFAULT_QUERY_SEED));
    }
    return rows;
  }

  public static void main(String[] args) {
    System.out.println("=== Phase 6B Commit 3: SQ8 Recall & Accuracy Benchmark Sweep ===");
    System.out.printf(
        "Configuration: D=%d, Metric=%s, k=%d, Queries=%d, DataSeed=%d, QuerySeed=%d\n\n",
        DEFAULT_DIMENSION,
        DEFAULT_METRIC,
        DEFAULT_K,
        DEFAULT_NUM_QUERIES,
        DEFAULT_DATA_SEED,
        DEFAULT_QUERY_SEED);

    System.out.println(
        "|          N | FP32 Flat Recall | SQ8 Scalar Recall | SQ8 SIMD Recall | Recall Loss | Scalar-SIMD Agreement |");
    System.out.println(
        "|-----------:|-----------------:|------------------:|----------------:|------------:|----------------------:|");

    List<RecallSweepRow> rows = runSweep();
    for (RecallSweepRow row : rows) {
      System.out.println(row.toMarkdownRow());
    }
  }
}
