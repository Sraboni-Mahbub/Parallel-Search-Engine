import pandas as pd
import time
import regex
import bisect
import threading
import ctypes

from concurrent.futures import ThreadPoolExecutor


# =========================================================
# GET CURRENT CPU / LOGICAL PROCESSOR
# Windows only
# =========================================================

def get_current_cpu():

    return ctypes.windll.kernel32.GetCurrentProcessorNumber()


# =========================================================
# 1. SEARCH ONE PARTITION
# =========================================================

def search_partition(pattern, start, end, documents):

    results = []

    # -----------------------------------------------------
    # Identify current worker thread
    # -----------------------------------------------------

    thread_name = threading.current_thread().name

    start_cpu = get_current_cpu()

    print(
        f"\n{thread_name} started"
        f" | Documents: {start} to {end - 1}"
        f" | Starting CPU: {start_cpu}"
    )


    # -----------------------------------------------------
    # Search assigned documents
    # -----------------------------------------------------

    for index in range(start, end):

        text = documents[index]

        # concurrent=True allows the regex library
        # to release the GIL during matching
        matches = pattern.findall(
            text,
            concurrent=True
        )

        score = len(matches)

        if score > 0:

            results.append(
                (index, score)
            )


    # -----------------------------------------------------
    # Check CPU again after search
    #
    # Windows may move a thread from one CPU to another.
    # -----------------------------------------------------

    end_cpu = get_current_cpu()

    print(
        f"{thread_name} finished"
        f" | Ending CPU: {end_cpu}"
        f" | Matches: {len(results)}"
    )

    return results


# =========================================================
# 2. SMART PARTITIONING
# =========================================================

def smart_partition(
    cumulative_workload,
    number_of_workers
):

    total_workload = cumulative_workload[-1]

    partitions = []

    workloads = []

    previous_index = 0
    previous_workload = 0


    # =====================================================
    # SMART PARTITIONING
    #
    # Divide total text workload approximately equally.
    # =====================================================

    for worker in range(
        1,
        number_of_workers
    ):

        target = (
            total_workload
            * worker
            / number_of_workers
        )

        boundary = bisect.bisect_left(
            cumulative_workload,
            target
        )

        partitions.append(
            (
                previous_index,
                boundary
            )
        )

        current_workload = (
            cumulative_workload[
                boundary - 1
            ]
            if boundary > 0
            else 0
        )

        workloads.append(
            current_workload
            - previous_workload
        )

        previous_index = boundary

        previous_workload = current_workload


    # =====================================================
    # LAST WORKER
    # =====================================================

    partitions.append(
        (
            previous_index,
            len(cumulative_workload)
        )
    )

    workloads.append(
        total_workload
        - previous_workload
    )


    return partitions, workloads


# =========================================================
# 3. MAIN PROGRAM
# =========================================================

if __name__ == "__main__":


    # =====================================================
    # LOAD DATASET
    # =====================================================

    file_path = "NewsCategorizer.csv"

    df = pd.read_csv(
        file_path
    )


    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    df["headline"] = (
        df["headline"].fillna("")
    )

    df["short_description"] = (
        df["short_description"].fillna("")
    )

    df["keywords"] = (
        df["keywords"].fillna("")
    )


    print(
        "Total documents:",
        len(df)
    )


    # =====================================================
    # PREPARE SEARCHABLE DOCUMENTS
    #
    # Lowercase ONCE before query
    # =====================================================

    documents = (
        df["headline"].astype(str)
        + " "
        + df["short_description"].astype(str)
        + " "
        + df["keywords"].astype(str)
    ).str.lower().tolist()


    # =====================================================
    # HETEROGENEOUS WORKLOAD
    #
    # Different documents have different lengths.
    # =====================================================

    document_workloads = [

        len(document)

        for document in documents
    ]


    # =====================================================
    # PRECOMPUTE CUMULATIVE WORKLOAD
    #
    # Allows partition boundaries to be found quickly.
    # =====================================================

    cumulative_workload = []

    total = 0

    for workload in document_workloads:

        total += workload

        cumulative_workload.append(
            total
        )


    # =====================================================
    # ENTER QUERY
    # =====================================================

    query = input(
        "\nEnter search query: "
    )


    # =====================================================
    # START TIMER
    # =====================================================

    start_time = time.perf_counter()


    # =====================================================
    # NUMBER OF WORKERS
    # =====================================================

    number_of_workers = 4


    # =====================================================
    # CREATE THREAD POOL
    # =====================================================

    executor = ThreadPoolExecutor(
        max_workers=number_of_workers
    )


    # =====================================================
    # SMART PARTITIONING
    #
    # INCLUDED IN TIMER
    # =====================================================

    partitions, workloads = smart_partition(
        cumulative_workload,
        number_of_workers
    )


    # =====================================================
    # COMPILE QUERY PATTERN ONCE
    # =====================================================

    query_words = (
        query.lower().split()
    )


    # Example:
    #
    # artificial intelligence
    #
    # becomes:
    #
    # \b(?:artificial|intelligence)\b

    search_pattern = (
        r"\b(?:"
        + "|".join(
            regex.escape(word)
            for word in query_words
        )
        + r")\b"
    )


    pattern = regex.compile(
        search_pattern
    )


    # =====================================================
    # PARALLEL SEARCH
    # =====================================================

    futures = []


    for start, end in partitions:

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
    # MERGE RESULTS
    # =====================================================

    results = []


    for future in futures:

        results.extend(
            future.result()
        )


    # =====================================================
    # SORT RESULTS
    # =====================================================

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # =====================================================
    # STOP TIMER
    # =====================================================

    end_time = time.perf_counter()


    execution_time = (
        end_time
        - start_time
    )


    # Worker shutdown outside timer
    executor.shutdown()


    # =====================================================
    # DISPLAY PARTITION INFORMATION
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "SMART PARTITIONING + "
        "HETEROGENEOUS WORKLOAD"
    )

    print(
        "========================================"
    )


    print(
        "Workers:",
        number_of_workers
    )


    for i, (start, end) in enumerate(
        partitions
    ):

        print(
            f"\nWorker {i + 1}"
        )

        print(
            "Documents:",
            end - start
        )

        print(
            "Document range:",
            start,
            "to",
            end - 1
        )

        print(
            "Workload:",
            workloads[i],
            "characters"
        )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

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
        len(documents)
    )

    print(
        "Matching documents:",
        len(results)
    )


    print(
        "\nTop 10 Results:"
    )


    for rank, (index, score) in enumerate(
        results[:10],
        start=1
    ):

        print(
            f"\n{rank}. "
            f"{df.iloc[index]['headline']}"
        )

        print(
            "Category:",
            df.iloc[index]["category"]
        )

        print(
            "Score:",
            score
        )


    # =====================================================
    # DISPLAY PARALLEL TIME
    # =====================================================

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
        f"Execution time: "
        f"{execution_time:.6f} seconds"
    )