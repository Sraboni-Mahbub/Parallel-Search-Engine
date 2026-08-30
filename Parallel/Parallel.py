# ============================================================
# Parallel.py
# Parallel Search Engine
# ============================================================

import pandas as pd
import time
import re
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from Partitioning import smart_partition


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "NewsCategorizer.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    df = pd.read_csv(
        DATASET_PATH
    )


    # --------------------------------------------------------
    # Replace missing values
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Combine searchable fields
    # --------------------------------------------------------

    documents = (
        df["headline"].astype(str)
        + " "
        + df["short_description"].astype(str)
        + " "
        + df["keywords"].astype(str)
    ).tolist()


    return df, documents


# ============================================================
# SEARCH ONE PARTITION
# ============================================================

def search_partition(args):
    """
    Search all documents assigned to one worker.
    """

    partition, query_words = args

    results = []


    # --------------------------------------------------------
    # Process every document in this partition
    # --------------------------------------------------------

    for index, document in partition:

        text = document.lower()

        score = 0


        # ----------------------------------------------------
        # Search every query word
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Store matching document
        # ----------------------------------------------------

        if score > 0:

            results.append(
                (
                    index,
                    score
                )
            )


    return results


# ============================================================
# PARALLEL SEARCH
# ============================================================

def parallel_search(
    query,
    documents,
    number_of_workers
):
    """
    Perform workload-aware parallel search.
    """

    # --------------------------------------------------------
    # Convert query to words
    # --------------------------------------------------------

    query_words = (
        query
        .lower()
        .split()
    )


    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query_words:

        return (
            [],
            0,
            [],
            []
        )


    # ========================================================
    # START TIMER
    # ========================================================

    start_time = time.perf_counter()


    # ========================================================
    # SMART PARTITIONING
    # ========================================================

    partitions, workloads = smart_partition(
        documents,
        query_words,
        number_of_workers
    )


    # ========================================================
    # CREATE TASKS
    # ========================================================

    tasks = []

    for partition in partitions:

        tasks.append(
            (
                partition,
                query_words
            )
        )


    # ========================================================
    # PARALLEL PROCESSING
    # ========================================================

    all_results = []


    with ProcessPoolExecutor(
        max_workers=number_of_workers
    ) as executor:

        worker_results = executor.map(
            search_partition,
            tasks
        )


        # ----------------------------------------------------
        # Collect results
        # ----------------------------------------------------

        for result in worker_results:

            all_results.extend(
                result
            )


    # ========================================================
    # MERGE RESULTS
    # ========================================================

    all_results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # ========================================================
    # STOP TIMER
    # ========================================================

    end_time = time.perf_counter()

    execution_time = (
        end_time - start_time
    )


    return (
        all_results,
        execution_time,
        partitions,
        workloads
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    df,
    results,
    query,
    execution_time,
    number_of_workers,
    partitions,
    workloads
):

    print("\n" + "=" * 70)

    print(
        "                 PARALLEL SEARCH ENGINE"
    )

    print("=" * 70)


    print(
        "Query:",
        query
    )

    print(
        "Documents searched:",
        len(df)
    )

    print(
        "Matching documents:",
        len(results)
    )

    print(
        "Number of workers:",
        number_of_workers
    )


    # ========================================================
    # WORKLOAD DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "             HETEROGENEOUS WORKLOAD"
    )

    print("=" * 70)


    total_workload = sum(
        workloads
    )


    for worker_index in range(
        number_of_workers
    ):

        worker_workload = (
            workloads[worker_index]
        )


        if total_workload > 0:

            percentage = (
                worker_workload
                / total_workload
                * 100
            )

        else:

            percentage = 0


        print(
            f"Worker {worker_index + 1}: "
            f"{len(partitions[worker_index])} documents | "
            f"Workload = {worker_workload} | "
            f"{percentage:.2f}%"
        )


    # ========================================================
    # TOP 10 RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "                    TOP 10 RESULTS"
    )

    print("=" * 70)


    for rank, (index, score) in enumerate(
        results[:10],
        start=1
    ):

        print(
            f"\n{rank}. "
            f"{df.iloc[index]['headline']}"
        )


        if "category" in df.columns:

            print(
                "Category:",
                df.iloc[index]["category"]
            )


        print(
            "Score:",
            score
        )


        print(
            "Description:",
            df.iloc[index]["short_description"]
        )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "             PARALLEL SEARCH PERFORMANCE"
    )

    print("=" * 70)


    print(
        f"Execution time: "
        f"{execution_time:.6f} seconds"
    )


    print("=" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # LOAD DATASET
    # ========================================================

    df, documents = load_dataset()


    print(
        "Total documents:",
        len(documents)
    )


    # ========================================================
    # CPU INFORMATION
    # ========================================================

    cpu_count = os.cpu_count()


    print(
        "Available CPU cores:",
        cpu_count
    )


    # ========================================================
    # USER QUERY
    # ========================================================

    query = input(
        "\nEnter search query: "
    ).strip()


    # ========================================================
    # WORKER SELECTION
    # ========================================================

    print(
        "\nAvailable worker configurations:"
    )


    # Don't recommend more workers than CPU cores

    available_workers = [
        worker
        for worker in [1, 2, 4, 8]
        if worker <= cpu_count
    ]


    for worker in available_workers:

        print(worker)


    try:

        number_of_workers = int(
            input(
                "\nEnter number of workers: "
            )
        )

    except ValueError:

        print(
            "Invalid input."
        )

        print(
            "Using 2 workers."
        )

        number_of_workers = 2


    # ========================================================
    # VALIDATE WORKER COUNT
    # ========================================================

    if number_of_workers < 1:

        number_of_workers = 1


    if number_of_workers > cpu_count:

        print(
            f"\nYour computer has only "
            f"{cpu_count} CPU cores."
        )

        print(
            f"Changing workers from "
            f"{number_of_workers} "
            f"to {cpu_count}."
        )

        number_of_workers = cpu_count


    if number_of_workers > len(documents):

        number_of_workers = len(
            documents
        )


    # ========================================================
    # RUN PARALLEL SEARCH
    # ========================================================

    (
        results,
        execution_time,
        partitions,
        workloads
    ) = parallel_search(
        query,
        documents,
        number_of_workers
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    display_results(
        df,
        results,
        query,
        execution_time,
        number_of_workers,
        partitions,
        workloads
    )