import bisect
import ctypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


# =========================================================
# REGEX ENGINE
# =========================================================

try:
    import regex as regex_engine

    SUPPORTS_CONCURRENT_REGEX = True

except ImportError:

    import re as regex_engine

    SUPPORTS_CONCURRENT_REGEX = False


# =========================================================
# NUMBER OF WORKERS
# =========================================================

# Exactly 4 worker threads.
WORKERS = 4


# =========================================================
# GET CURRENT CPU
# =========================================================

def get_current_cpu():
    """
    Return the logical CPU currently executing
    this thread on Windows.
    """

    try:

        return (
            ctypes.windll.kernel32
            .GetCurrentProcessorNumber()
        )

    except (AttributeError, OSError):

        return "N/A"


# =========================================================
# SEARCH ONE PARTITION
# =========================================================

def search_partition(
    pattern,
    start,
    end,
    documents
):

    results = []

    thread_name = (
        threading
        .current_thread()
        .name
    )

    starting_cpu = (
        get_current_cpu()
    )


    print(

        f"\n{thread_name} started"

        f" | Rows: {start} to {end - 1}"

        f" | Starting CPU: {starting_cpu}"

    )


    # Local variables reduce repeated lookups.

    local_documents = documents

    local_append = results.append

    local_findall = pattern.findall


    # =====================================================
    # THIRD-PARTY REGEX
    # =====================================================

    if SUPPORTS_CONCURRENT_REGEX:

        for index in range(
            start,
            end
        ):

            score = len(

                local_findall(

                    local_documents[index],

                    concurrent=True

                )

            )


            if score > 0:

                local_append(
                    (index, score)
                )


    # =====================================================
    # BUILT-IN RE FALLBACK
    # =====================================================

    else:

        for index in range(
            start,
            end
        ):

            score = len(

                local_findall(
                    local_documents[index]
                )

            )


            if score > 0:

                local_append(
                    (index, score)
                )


    ending_cpu = (
        get_current_cpu()
    )


    print(

        f"{thread_name} finished"

        f" | Ending CPU: {ending_cpu}"

        f" | Matches: {len(results)}"

    )


    return results


# =========================================================
# SMART PARTITIONING
# =========================================================

def smart_partition(
    cumulative_workload,
    number_of_workers
):

    if not cumulative_workload:

        return [], []


    total_workload = (
        cumulative_workload[-1]
    )


    partitions = []

    workloads = []


    previous_index = 0

    previous_workload = 0


    # =====================================================
    # CREATE BALANCED PARTITIONS
    # =====================================================

    for worker in range(
        1,
        number_of_workers
    ):


        target_workload = (

            total_workload

            * worker

            / number_of_workers

        )


        boundary = bisect.bisect_left(

            cumulative_workload,

            target_workload

        )


        if boundary > 0:

            current_workload = (

                cumulative_workload[
                    boundary - 1
                ]

            )

        else:

            current_workload = 0


        partitions.append(

            (
                previous_index,
                boundary
            )

        )


        workloads.append(

            current_workload

            - previous_workload

        )


        previous_index = boundary

        previous_workload = (
            current_workload
        )


    # =====================================================
    # FINAL PARTITION
    # =====================================================

    partitions.append(

        (
            previous_index,

            len(
                cumulative_workload
            )
        )

    )


    workloads.append(

        total_workload

        - previous_workload

    )


    return (
        partitions,
        workloads
    )


# =========================================================
# LOAD DATASET
# =========================================================

def load_documents(
    file_path
):

    if not file_path.exists():

        raise FileNotFoundError(

            f"Dataset was not found:\n"

            f"{file_path}"

        )


    df = pd.read_csv(
        file_path
    )


    required_columns = {

        "headline",

        "short_description",

        "category"

    }


    missing_columns = (

        required_columns.difference(
            df.columns
        )

    )


    if missing_columns:

        raise ValueError(

            "Missing columns: "

            + ", ".join(
                sorted(
                    missing_columns
                )
            )

        )


    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    df["headline"] = (

        df["headline"]
        .fillna("")

    )


    df["short_description"] = (

        df["short_description"]
        .fillna("")

    )


    df["category"] = (

        df["category"]
        .fillna("")

    )


    # =====================================================
    # PREPARE DOCUMENTS
    # =====================================================

    documents = (

        df["headline"]
        .astype(str)

        + " "

        + df[
            "short_description"
        ].astype(str)

        + " "

        + df[
            "category"
        ].astype(str)

    ).str.lower().tolist()


    return (
        df,
        documents
    )


# =========================================================
# CALCULATE WORKLOAD
# =========================================================

def build_cumulative_workload(
    documents
):

    cumulative_workload = []

    total = 0


    for document in documents:

        total += len(
            document
        )

        cumulative_workload.append(
            total
        )


    return cumulative_workload


# =========================================================
# CREATE SEARCH PATTERN
# =========================================================

def build_search_pattern(
    query
):

    query_words = (

        query
        .lower()
        .split()

    )


    if not query_words:

        raise ValueError(
            "Search query cannot be empty."
        )


    search_pattern = (

        r"\b(?:"

        + "|".join(

            regex_engine.escape(
                word
            )

            for word in query_words

        )

        + r")\b"

    )


    pattern = regex_engine.compile(

        search_pattern,

        regex_engine.IGNORECASE

    )


    return pattern


# =========================================================
# PARALLEL SEARCH
# =========================================================

def parallel_search(
    pattern,
    documents,
    cumulative_workload,
    executor
):

    # =====================================================
    # START TIMER
    # =====================================================

    start_time = (
        time.perf_counter()
    )


    # =====================================================
    # SMART PARTITION INTO EXACTLY 4 PARTS
    # =====================================================

    partitions, workloads = (

        smart_partition(

            cumulative_workload,

            WORKERS

        )

    )


    futures = []


    # =====================================================
    # DISTRIBUTE WORK
    # =====================================================

    for start, end in partitions:


        if start >= end:

            continue


        future = executor.submit(

            search_partition,

            pattern,

            start,

            end,

            documents

        )


        futures.append(
            future
        )


    # =====================================================
    # COLLECT RESULTS
    # =====================================================

    results = []


    for future in futures:

        worker_results = (
            future.result()
        )

        results.extend(
            worker_results
        )


    # =====================================================
    # SORT RESULTS
    # =====================================================

    results.sort(

        key=lambda item: item[1],

        reverse=True

    )


    # =====================================================
    # STOP TIMER
    # =====================================================

    end_time = (
        time.perf_counter()
    )


    execution_time = (

        end_time

        - start_time

    )


    return (

        results,

        execution_time,

        partitions,

        workloads

    )


# =========================================================
# DISPLAY PARTITION INFORMATION
# =========================================================

def display_partition_information(
    partitions,
    workloads
):

    print(
        "\n========================================"
    )

    print(
        "SMART PARTITION INFORMATION"
    )

    print(
        "========================================"
    )


    print(
        "Workers:",
        WORKERS
    )


    for worker_number, (
        (start, end),
        workload
    ) in enumerate(

        zip(
            partitions,
            workloads
        ),

        start=1

    ):


        print(
            f"\nWorker {worker_number}"
        )


        print(

            "Document range:",

            start,

            "to",

            end - 1

        )


        print(

            "Number of documents:",

            end - start

        )


        print(

            "Workload:",

            f"{workload:,}",

            "characters"

        )


# =========================================================
# DISPLAY SEARCH RESULTS
# =========================================================

def display_search_results(
    df,
    query,
    results
):

    print(
        "\n========================================"
    )

    print(
        "PARALLEL SEARCH RESULTS"
    )

    print(
        "========================================"
    )


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
        "\nTop 10 Results:"
    )


    if not results:

        print(
            "No matching documents found."
        )

        return


    for rank, (
        index,
        score
    ) in enumerate(

        results[:10],

        start=1

    ):


        row = (
            df.iloc[index]
        )


        print(

            f"\n{rank}. "

            f"{row['headline']}"

        )


        print(

            "Category:",

            row["category"]

        )


        print(

            "Score:",

            score

        )


# =========================================================
# MAIN
# =========================================================

def main():


    # CSV file should be in the same
    # Dataset2 folder as this script.

    file_path = (

        Path(__file__).parent

        / "News_Category_Dataset_v3.csv"

    )


    # =====================================================
    # LOAD DATA
    # NOT TIMED
    # =====================================================

    df, documents = (

        load_documents(
            file_path
        )

    )


    # =====================================================
    # PRECOMPUTE WORKLOAD
    # NOT TIMED
    # =====================================================

    cumulative_workload = (

        build_cumulative_workload(
            documents
        )

    )


    print(
        "Dataset:",
        file_path
    )


    print(
        "Total documents:",
        len(df)
    )


    print(
        "Worker threads used:",
        WORKERS
    )


    if SUPPORTS_CONCURRENT_REGEX:

        print(
            "Regex engine: regex "
            "(concurrent matching enabled)"
        )

    else:

        print(
            "Regex engine: built-in re"
        )

        print(
            "For better threading performance:"
        )

        print(
            "pip install regex"
        )


    # =====================================================
    # USER INPUT
    # NOT TIMED
    # =====================================================

    query = input(
        "\nEnter search query: "
    )


    # =====================================================
    # COMPILE REGEX
    # NOT TIMED
    # =====================================================

    pattern = (
        build_search_pattern(
            query
        )
    )


    # =====================================================
    # CREATE EXACTLY 4 THREADS
    # NOT TIMED
    # =====================================================

    with ThreadPoolExecutor(

        max_workers=WORKERS,

        thread_name_prefix="SearchWorker"

    ) as executor:


        # =================================================
        # RUN SEARCH
        # =================================================

        (
            results,

            execution_time,

            partitions,

            workloads

        ) = parallel_search(

            pattern,

            documents,

            cumulative_workload,

            executor

        )


    # =====================================================
    # DISPLAY
    # =====================================================

    display_partition_information(

        partitions,

        workloads

    )


    display_search_results(

        df,

        query,

        results

    )


    print(
        "\n========================================"
    )

    print(
        "PARALLEL SEARCH PERFORMANCE"
    )

    print(
        "========================================"
    )


    print(
        "Worker threads:",
        WORKERS
    )


    print(

        f"Execution time: "

        f"{execution_time:.6f} seconds"

    )


# =========================================================
# START PROGRAM
# =========================================================

if __name__ == "__main__":

    main()