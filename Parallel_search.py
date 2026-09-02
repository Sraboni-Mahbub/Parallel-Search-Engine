import pandas as pd
import time
import re
import heapq
import multiprocessing as mp


# =========================================================
# 1. SEARCH FUNCTION FOR EACH WORKER
# =========================================================

def search_partition(task):

    query, partition = task

    query_words = query.lower().split()

    results = []

    # Each worker searches only its assigned partition
    for index, document in partition:

        text = document.lower()

        score = 0

        # Same search logic as sequential version
        for word in query_words:

            occurrences = len(
                re.findall(
                    r"\b" + re.escape(word) + r"\b",
                    text
                )
            )

            score += occurrences

        # Save matching document
        if score > 0:
            results.append(
                (index, score)
            )

    return results


# =========================================================
# 2. SMART PARTITIONING
# =========================================================

def smart_partition(documents, number_of_workers):

    # Create one partition for each worker
    partitions = [
        [] for _ in range(number_of_workers)
    ]

    # Store current workload of each worker
    workloads = [
        0 for _ in range(number_of_workers)
    ]

    # -----------------------------------------------------
    # Heap contains:
    #
    # (current workload, worker number)
    #
    # The worker with the smallest workload
    # will always be selected first.
    # -----------------------------------------------------

    worker_heap = [
        (0, worker)
        for worker in range(number_of_workers)
    ]

    heapq.heapify(worker_heap)


    # =====================================================
    # HETEROGENEOUS WORKLOAD
    # =====================================================
    #
    # Different documents have different lengths.
    #
    # Longer document = heavier workload
    # Shorter document = lighter workload
    #
    # Smart partitioning distributes these different
    # workloads among the workers.
    # =====================================================

    for index, document in enumerate(documents):

        # Get worker with smallest current workload
        current_load, worker = heapq.heappop(
            worker_heap
        )

        # Assign this document to that worker
        partitions[worker].append(
            (index, document)
        )

        # Update worker workload
        new_load = (
            current_load
            + len(document)
        )

        workloads[worker] = new_load

        # Put worker back into heap
        heapq.heappush(
            worker_heap,
            (new_load, worker)
        )

    return partitions, workloads


# =========================================================
# 3. MAIN PROGRAM
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


    # Replace missing values with empty strings
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
    # =====================================================

    documents = (
        df["headline"].astype(str)
        + " "
        + df["short_description"].astype(str)
        + " "
        + df["keywords"].astype(str)
    ).tolist()


    # =====================================================
    # ENTER QUERY
    # =====================================================

    query = input(
        "\nEnter search query: "
    )


    # =====================================================
    # START TOTAL TIMER
    #
    # Timer starts immediately after user presses Enter.
    # =====================================================

    start_time = time.perf_counter()


    # =====================================================
    # NUMBER OF WORKERS
    # =====================================================

    number_of_workers = 4


    # =====================================================
    # CREATE 4 PARALLEL WORKERS
    #
    # This is INCLUDED in total execution time.
    # =====================================================

    pool = mp.Pool(
        processes=number_of_workers
    )


    # =====================================================
    # SMART PARTITIONING
    #
    # This is INCLUDED in total execution time.
    # =====================================================

    partitions, workloads = smart_partition(
        documents,
        number_of_workers
    )


    # =====================================================
    # DISPLAY PARTITION INFORMATION
    # =====================================================

    print(
        "\nSMART PARTITIONING + "
        "HETEROGENEOUS WORKLOAD"
    )

    print(
        "Workers:",
        number_of_workers
    )


    for i in range(number_of_workers):

        print(
            f"\nWorker {i + 1}"
        )

        print(
            "Documents:",
            len(partitions[i])
        )

        print(
            "Workload:",
            workloads[i],
            "characters"
        )


    # =====================================================
    # PREPARE TASKS
    # =====================================================

    tasks = []

    for i in range(number_of_workers):

        tasks.append(
            (
                query,
                partitions[i]
            )
        )


    # =====================================================
    # PARALLEL SEARCH
    #
    # All 4 workers search simultaneously.
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
    # Same sorting logic as Sequential.py
    # =====================================================

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print(
        "\nSEARCH RESULTS"
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
        "\nTop 10 Results : "
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
    # END TOTAL TIMER
    #
    # Timer stops after results are displayed.
    # =====================================================

    end_time = time.perf_counter()


    # =====================================================
    # CALCULATE EXECUTION TIME
    # =====================================================

    execution_time = (
        end_time
        - start_time
    )


    # =====================================================
    # DISPLAY TOTAL TIME
    # =====================================================

    print(
        "\nTIME CALCULATION"
    )

    print(
        f"Total parallel execution time: "
        f"{execution_time:.6f} seconds"
    )


    # =====================================================
    # CLOSE WORKER PROCESSES
    # =====================================================

    pool.close()

    pool.join()