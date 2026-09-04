import os
import time
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


# ============================================================
# NUMBER OF WORKERS
# ============================================================

# Your PC has 4 physical cores.
# We deliberately use exactly 4 worker processes.
WORKERS = 4


# Each process keeps access to the prepared documents.
WORKER_DOCUMENTS = None


# ============================================================
# INITIALIZE EACH PROCESS
# ============================================================

def initialize_worker(documents):
    """
    Store the documents inside each worker process.
    This happens before the search timer starts.
    """

    global WORKER_DOCUMENTS
    WORKER_DOCUMENTS = documents


# ============================================================
# WARM-UP FUNCTION
# ============================================================

def warmup_worker(x):
    """
    Small operation used to start worker processes
    before timing begins.
    """

    value = 0

    for i in range(10000):
        value += (i * i) % 97

    return os.getpid(), value + x


# ============================================================
# SEARCH ONE PARTITION
# ============================================================

def search_range(
    start_index,
    end_index,
    query_words
):
    """
    Search only the document range assigned
    to this worker.
    """

    local_results = []

    # Compile regex once for this worker task.
    patterns = [
        re.compile(
            r"\b"
            + re.escape(word)
            + r"\b"
        )
        for word in query_words
    ]

    documents = WORKER_DOCUMENTS

    for index in range(
        start_index,
        end_index
    ):

        text = documents[index]

        score = 0

        for pattern in patterns:

            occurrences = len(
                pattern.findall(text)
            )

            score += occurrences

        if score > 0:

            local_results.append(
                (index, score)
            )

    return local_results


# ============================================================
# PARALLEL SEARCH
# ============================================================

def parallel_search(
    query_words,
    total_documents,
    executor
):

    # ========================================================
    # TIMER START
    # ========================================================

    start_time = time.perf_counter()


    # ========================================================
    # DIVIDE DATA INTO EXACTLY 4 PARTS
    # ========================================================

    chunk_size = (
        total_documents
        + WORKERS
        - 1
    ) // WORKERS


    futures = []


    # ========================================================
    # ASSIGN ONE PART TO EACH WORKER
    # ========================================================

    for worker_id in range(WORKERS):

        start_index = (
            worker_id
            * chunk_size
        )

        end_index = min(
            start_index + chunk_size,
            total_documents
        )

        if start_index >= end_index:
            break

        future = executor.submit(
            search_range,
            start_index,
            end_index,
            query_words
        )

        futures.append(future)


    # ========================================================
    # COLLECT RESULTS
    # ========================================================

    results = []

    for future in futures:

        worker_results = future.result()

        results.extend(
            worker_results
        )


    # ========================================================
    # SORT RESULTS
    # ========================================================

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # ========================================================
    # TIMER END
    # ========================================================

    end_time = time.perf_counter()

    execution_time = (
        end_time
        - start_time
    )

    return (
        results,
        execution_time
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # CSV should be in the same Dataset2 folder
    # as this Python file.

    file_path = (
        Path(__file__).parent
        / "News_Category_Dataset_v3.csv"
    )


    # ========================================================
    # LOAD DATASET
    # NOT TIMED
    # ========================================================

    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )


    df = pd.read_csv(
        file_path
    )


    # ========================================================
    # HANDLE MISSING VALUES
    # ========================================================

    df["headline"] = (
        df["headline"].fillna("")
    )

    df["short_description"] = (
        df["short_description"].fillna("")
    )


    print(
        "Total documents:",
        len(df)
    )

    print(
        "Worker processes used:",
        WORKERS
    )


    # ========================================================
    # PREPARE DOCUMENTS
    # NOT TIMED
    # ========================================================

    documents = (

        df["headline"].astype(str)

        + " "

        + df[
            "short_description"
        ].astype(str)

    ).str.lower().tolist()


    # ========================================================
    # USER INPUT
    # NOT TIMED
    # ========================================================

    query = input(
        "\nEnter search query: "
    )


    query_words = (
        query
        .lower()
        .split()
    )


    if not query_words:

        print(
            "Please enter at least one search word."
        )

        return


    # ========================================================
    # CREATE EXACTLY 4 PROCESSES
    # NOT TIMED
    # ========================================================

    with ProcessPoolExecutor(

        max_workers=WORKERS,

        initializer=initialize_worker,

        initargs=(documents,)

    ) as executor:


        # ====================================================
        # WARM UP
        # NOT TIMED
        # ====================================================

        warmup_results = list(

            executor.map(
                warmup_worker,
                range(WORKERS)
            )

        )


        unique_pids = {

            pid

            for pid, value
            in warmup_results

        }


        print(
            "Worker processes:",
            WORKERS
        )


        # ====================================================
        # PARALLEL SEARCH
        # ====================================================

        results, execution_time = (

            parallel_search(
                query_words,
                len(documents),
                executor
            )

        )


    # ========================================================
    # DISPLAY RESULTS
    # NOT TIMED
    # ========================================================

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


    for rank, (
        index,
        score
    ) in enumerate(
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


    # ========================================================
    # PERFORMANCE
    # ========================================================

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
        "Workers:",
        WORKERS
    )

    print(
        f"Execution time: "
        f"{execution_time:.6f} seconds"
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()