import pandas as pd
import time
import re
import bisect
import multiprocessing as mp


# =========================================================
# GLOBAL DOCUMENT LIST FOR EACH CPU PROCESS
# =========================================================

worker_documents = None


# =========================================================
# 1. INITIALIZE CPU WORKERS
# =========================================================

def initialize_worker(documents):

    global worker_documents

    # Each process receives the documents once
    # when the search engine starts.
    worker_documents = documents


# =========================================================
# 2. SEARCH ONE PARTITION
# =========================================================

def search_partition(task):

    query, start, end = task

    query_words = query.lower().split()

    # Compile query patterns once for this worker
    patterns = [
        re.compile(
            r"\b" + re.escape(word) + r"\b"
        )
        for word in query_words
    ]

    results = []


    # Each CPU process searches only its own range
    for index in range(start, end):

        # Documents are already lowercase
        text = worker_documents[index]

        score = 0


        for pattern in patterns:

            occurrences = len(
                pattern.findall(text)
            )

            score += occurrences


        if score > 0:

            results.append(
                (index, score)
            )


    return results


# =========================================================
# 3. SMART PARTITIONING
# =========================================================

def smart_partition(
    cumulative_workload,
    number_of_workers
):

    total_workload = (
        cumulative_workload[-1]
    )


    partitions = []

    workloads = []


    previous_index = 0

    previous_workload = 0


    # =====================================================
    # HETEROGENEOUS WORKLOAD
    #
    # Documents do not all contain the same amount of text.
    #
    # Therefore:
    #
    # Longer document = heavier workload
    # Shorter document = lighter workload
    #
    # Smart partitioning divides TOTAL TEXT WORKLOAD,
    # instead of simply dividing document count.
    # =====================================================


    for worker in range(
        1,
        number_of_workers
    ):

        # Workload this worker boundary should reach
        target = (
            total_workload
            * worker
            / number_of_workers
        )


        # Find document index nearest to target workload
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

        previous_workload = (
            current_workload
        )


    # =====================================================
    # LAST PROCESS
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
# 4. MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    # Required for Windows multiprocessing
    mp.freeze_support()


    # =====================================================
    # LOAD DATASET
    # =====================================================

    file_path = "NewsCategorizer.csv"


    df = pd.read_csv(
        file_path
    )


    # Replace missing values
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
    # Lowercase ONCE.
    #
    # We do NOT lowercase 50,000 documents during
    # every search.
    # =====================================================

    documents = (
        df["headline"].astype(str)
        + " "
        + df["short_description"].astype(str)
        + " "
        + df["keywords"].astype(str)
    ).str.lower().tolist()


    # =====================================================
    # CALCULATE HETEROGENEOUS DOCUMENT WORKLOAD
    #
    # workload = number of characters in document
    # =====================================================

    document_workloads = [
        len(document)
        for document in documents
    ]


    # =====================================================
    # PRECOMPUTE CUMULATIVE WORKLOAD
    #
    # This allows smart partitioning to happen extremely
    # quickly after the query is entered.
    # =====================================================

    cumulative_workload = []

    total = 0


    for workload in document_workloads:

        total += workload

        cumulative_workload.append(
            total
        )


    # =====================================================
    # CPU INFORMATION
    # =====================================================

    available_cpus = (
        mp.cpu_count()
    )


    print(
        "\nLogical CPU processors available:",
        available_cpus
    )


    # =====================================================
    # NUMBER OF CPU PROCESSES
    #
    # Start with 4 CPU processes.
    #
    # This still uses actual CPU multiprocessing.
    #
    # You can later test 2, 3, 4 etc. to find
    # the optimal number for your computer.
    # =====================================================

    number_of_workers = min(
        4,
        available_cpus
    )


    print(
        "CPU processes being used:",
        number_of_workers
    )


    # =====================================================
    # START PARALLEL SEARCH ENGINE
    #
    # IMPORTANT:
    #
    # CPU processes are created ONCE and kept alive.
    #
    # This prevents Windows from spending several seconds
    # creating processes every time the user searches.
    # =====================================================

    print(
        "\nInitializing CPU workers..."
    )


    pool = mp.Pool(

        processes=number_of_workers,

        initializer=initialize_worker,

        initargs=(documents,)
    )


    print(
        "Parallel search engine ready."
    )


    # =====================================================
    # ENTER QUERY
    # =====================================================

    query = input(
        "\nEnter search query: "
    )


    # =====================================================
    # START SEARCH TIMER
    #
    # TIMER STARTS IMMEDIATELY AFTER USER PRESSES ENTER
    # =====================================================

    start_time = time.perf_counter()


    # =====================================================
    # SMART PARTITIONING
    #
    # INCLUDED IN SEARCH TIME
    # =====================================================

    partitions, workloads = (
        smart_partition(
            cumulative_workload,
            number_of_workers
        )
    )


    # =====================================================
    # PREPARE SMALL CPU TASKS
    #
    # Only query + start index + end index are sent.
    #
    # We DO NOT send thousands of document strings.
    # =====================================================

    tasks = [
        (
            query,
            start,
            end
        )

        for start, end in partitions
    ]


    # =====================================================
    # PARALLEL CPU SEARCH
    #
    # Different processes can execute on different
    # processor cores simultaneously.
    # =====================================================

    worker_results = pool.map(
        search_partition,
        tasks
    )


    # =====================================================
    # MERGE RESULTS
    # =====================================================

    results = []


    for worker_result in worker_results:

        results.extend(
            worker_result
        )


    # =====================================================
    # SORT RESULTS
    #
    # Same ranking principle as Sequential.py
    # =====================================================

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # =====================================================
    # STOP SEARCH TIMER
    #
    # Results are now completely ready.
    # =====================================================

    end_time = time.perf_counter()


    execution_time = (
        end_time
        - start_time
    )


    # =====================================================
    # DISPLAY SMART PARTITION INFORMATION
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


    for i, (start, end) in enumerate(
        partitions
    ):

        print(
            f"\nCPU Process {i + 1}"
        )


        print(
            "Documents:",
            end - start
        )


        print(
            "Workload:",
            workloads[i],
            "characters"
        )


    # =====================================================
    # DISPLAY SEARCH RESULTS
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


    # =====================================================
    # DISPLAY PARALLEL SEARCH TIME
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


    # =====================================================
    # CLOSE CPU PROCESSES
    # =====================================================

    pool.close()

    pool.join()