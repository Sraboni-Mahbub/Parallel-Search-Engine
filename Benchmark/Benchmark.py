# ============================================================
# Benchmark.py
# Multi-Query Performance Benchmark
# Sequential vs Parallel Search
# ============================================================

import sys
import time
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "NewsCategorizer.csv"
)

PARALLEL_DIR = (
    BASE_DIR
    / "Parallel"
)

# Allow Python to import Parallel.py
sys.path.insert(
    0,
    str(PARALLEL_DIR)
)

from Parallel import parallel_search


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    df = pd.read_csv(
        DATASET_PATH
    )

    df["headline"] = (
        df["headline"]
        .fillna("")
    )

    df["short_description"] = (
        df["short_description"]
        .fillna("")
    )

    df["keywords"] = (
        df["keywords"]
        .fillna("")
    )

    documents = (
        df["headline"].astype(str)
        + " "
        + df["short_description"].astype(str)
        + " "
        + df["keywords"].astype(str)
    ).tolist()

    return df, documents


# ============================================================
# SEQUENTIAL SEARCH
# ============================================================

def sequential_search(
    query,
    documents
):

    import re

    query_words = (
        query.lower()
        .split()
    )

    results = []

    start_time = time.perf_counter()

    for index, document in enumerate(
        documents
    ):

        text = document.lower()

        score = 0

        for word in query_words:

            occurrences = len(
                re.findall(
                    r"\b"
                    + re.escape(word)
                    + r"\b",
                    text
                )
            )

            score += occurrences

        if score > 0:

            results.append(
                (
                    index,
                    score
                )
            )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    end_time = time.perf_counter()

    return (
        results,
        end_time - start_time
    )


# ============================================================
# MAIN BENCHMARK
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print(
        "          MULTI-QUERY PARALLEL SEARCH BENCHMARK"
    )
    print("=" * 80)


    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("\nLoading dataset...")

    df, documents = load_dataset()

    print(
        "Total documents:",
        len(documents)
    )


    # ========================================================
    # 20 SEARCH QUERIES
    # ========================================================

    queries = [
        "sports",
        "politics",
        "technology",
        "health",
        "business",
        "education",
        "travel",
        "food",
        "music",
        "government",
        "economy",
        "family",
        "children",
        "movie",
        "football",
        "fashion",
        "science",
        "world",
        "life",
        "news"
    ]


    print("\nBenchmark queries:")

    for number, query in enumerate(
        queries,
        start=1
    ):

        print(
            f"{number:2}. {query}"
        )


    print(
        "\nTotal queries:",
        len(queries)
    )


    # ========================================================
    # WORKER CONFIGURATIONS
    # ========================================================

    worker_counts = [
        1,
        2,
        4
    ]


    # ========================================================
    # SEQUENTIAL BENCHMARK
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "              SEQUENTIAL BENCHMARK"
    )
    print("=" * 80)


    sequential_total_time = 0

    sequential_query_results = {}


    benchmark_start = time.perf_counter()


    for query in queries:

        results, execution_time = (
            sequential_search(
                query,
                documents
            )
        )

        sequential_total_time += (
            execution_time
        )

        sequential_query_results[
            query
        ] = results


        print(
            f"{query:<15} "
            f"{execution_time:.6f} seconds | "
            f"Results: {len(results)}"
        )


    benchmark_end = time.perf_counter()


    print("\n" + "-" * 80)

    print(
        f"Total sequential search time: "
        f"{sequential_total_time:.6f} seconds"
    )


    # ========================================================
    # PARALLEL BENCHMARK
    # ========================================================

    all_benchmark_results = []


    print("\n")
    print("=" * 80)
    print(
        "               PARALLEL BENCHMARK"
    )
    print("=" * 80)


    for workers in worker_counts:

        print("\n")
        print("-" * 80)

        print(
            f"Testing {workers} worker(s)"
        )

        print("-" * 80)


        parallel_total_time = 0

        total_matching_documents = 0

        all_results_correct = True


        # ====================================================
        # RUN ALL 20 QUERIES
        # ====================================================

        for query in queries:

            (
                parallel_results,
                execution_time,
                partitions,
                workloads
            ) = parallel_search(
                query,
                documents,
                workers
            )


            parallel_total_time += (
                execution_time
            )


            total_matching_documents += (
                len(parallel_results)
            )


            # -----------------------------------------------
            # Check correctness
            # -----------------------------------------------

            sequential_results = (
                sequential_query_results[
                    query
                ]
            )


            sequential_indexes = sorted(
                index
                for index, score
                in sequential_results
            )


            parallel_indexes = sorted(
                index
                for index, score
                in parallel_results
            )


            if (
                sequential_indexes
                != parallel_indexes
            ):

                all_results_correct = False


            print(
                f"{query:<15} "
                f"{execution_time:.6f} seconds | "
                f"Results: {len(parallel_results)}"
            )


        # ====================================================
        # SPEEDUP
        # ====================================================

        speedup = (
            sequential_total_time
            / parallel_total_time
        )


        # ====================================================
        # EFFICIENCY
        # ====================================================

        efficiency = (
            speedup
            / workers
            * 100
        )


        # ====================================================
        # STORE RESULT
        # ====================================================

        all_benchmark_results.append(
            {
                "Workers": workers,
                "Sequential_Time": sequential_total_time,
                "Parallel_Time": parallel_total_time,
                "Speedup": speedup,
                "Efficiency_Percent": efficiency,
                "Results_Correct": all_results_correct
            }
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        print("\n")

        print(
            f"Total parallel time: "
            f"{parallel_total_time:.6f} seconds"
        )

        print(
            f"Speedup: "
            f"{speedup:.3f}x"
        )

        print(
            f"Efficiency: "
            f"{efficiency:.2f}%"
        )

        print(
            f"Results correct: "
            f"{all_results_correct}"
        )


    # ========================================================
    # FINAL DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_benchmark_results
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "                 FINAL PERFORMANCE RESULTS"
    )

    print("=" * 80)


    print(
        f"{'Workers':<10}"
        f"{'Sequential':<18}"
        f"{'Parallel':<18}"
        f"{'Speedup':<15}"
        f"{'Efficiency':<15}"
    )

    print("-" * 80)


    for _, row in results_df.iterrows():

        print(
            f"{int(row['Workers']):<10}"
            f"{row['Sequential_Time']:<18.6f}"
            f"{row['Parallel_Time']:<18.6f}"
            f"{row['Speedup']:<15.3f}"
            f"{row['Efficiency_Percent']:<14.2f}%"
        )


    print("=" * 80)


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_dir = (
        BASE_DIR
        / "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )


    output_file = (
        results_dir
        / "performance_results.csv"
    )


    results_df.to_csv(
        output_file,
        index=False
    )


    print(
        "\nResults saved to:"
    )

    print(
        output_file
    )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "                 BENCHMARK COMPLETE"
    )

    print("=" * 80)
    