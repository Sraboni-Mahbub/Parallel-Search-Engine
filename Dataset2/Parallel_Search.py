import os
import re
import time
from pathlib import Path
from concurrent.futures import (
    ProcessPoolExecutor,
    wait,
    FIRST_COMPLETED
)
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent
FILE_PATH = BASE_DIR / "News_Category_Dataset_v3.csv"

WORKER_DOCUMENTS = None


# ============================================================
# WORKER INITIALIZATION
# ============================================================

def initialize_worker(documents):
    """
    Each worker receives the document collection once.
    """
    global WORKER_DOCUMENTS
    WORKER_DOCUMENTS = documents


# ============================================================
# WARM-UP
# ============================================================

def warmup_worker(_):
    """
    Small warm-up operation so worker processes are ready
    before the actual search timing begins.
    """
    total = 0

    for i in range(10000):
        total += i

    return os.getpid()


# ============================================================
# SEARCH ONE CHUNK
# ============================================================

def search_chunk(start_index, end_index, query_words):
    """
    Searches a small portion of the document collection.

    Returns:
        List of (document_index, score)
    """

    results = []

    # Compile regex once for this chunk
    patterns = [
        re.compile(r"\b" + re.escape(word) + r"\b")
        for word in query_words
    ]

    for index in range(start_index, end_index):

        text = WORKER_DOCUMENTS[index]

        score = 0

        for pattern in patterns:
            score += len(pattern.findall(text))

        if score > 0:
            results.append((index, score))

    return results


# ============================================================
# SMART / DYNAMIC PARTITIONING
# ============================================================

def create_smart_chunks(total_documents, workers):
    """
    Creates many smaller chunks instead of one chunk per worker.

    This allows workers to receive new work whenever they finish
    their current chunk.

    More chunks than workers = better load balancing.
    """

    # Create approximately 8 chunks per worker.
    # Minimum chunk size prevents creating thousands of tiny tasks.
    chunk_size = max(
        1000,
        total_documents // (workers * 8)
    )

    chunks = []

    start = 0

    while start < total_documents:

        end = min(
            start + chunk_size,
            total_documents
        )

        chunks.append((start, end))

        start = end

    return chunks


# ============================================================
# PARALLEL SEARCH
# ============================================================

def parallel_search(query_words, total_documents, executor, workers):
    """
    Performs parallel search using dynamic smart partitioning.

    At most 'workers' chunks are active at the same time.
    When one worker finishes, another chunk is submitted.
    """

    # --------------------------------------------------------
    # Create many small chunks
    # --------------------------------------------------------

    chunks = create_smart_chunks(
        total_documents,
        workers
    )

    print("\nSmart Partitioning")
    print("------------------")
    print(f"Total documents : {total_documents}")
    print(f"Workers         : {workers}")
    print(f"Total chunks    : {len(chunks)}")

    # --------------------------------------------------------
    # Start timing
    # --------------------------------------------------------

    start_time = time.perf_counter()

    all_results = []

    # --------------------------------------------------------
    # Submit initial chunks
    # --------------------------------------------------------

    next_chunk = 0

    active_futures = {}

    initial_tasks = min(
        workers,
        len(chunks)
    )

    for _ in range(initial_tasks):

        start_index, end_index = chunks[next_chunk]

        future = executor.submit(
            search_chunk,
            start_index,
            end_index,
            query_words
        )

        active_futures[future] = (
            start_index,
            end_index
        )

        next_chunk += 1

    # --------------------------------------------------------
    # Dynamic scheduling
    # --------------------------------------------------------

    while active_futures:

        completed, _ = wait(
            active_futures,
            return_when=FIRST_COMPLETED
        )

        for future in completed:

            chunk_info = active_futures.pop(future)

            chunk_results = future.result()

            all_results.extend(chunk_results)

            # ------------------------------------------------
            # Immediately give another chunk to the
            # available worker
            # ------------------------------------------------

            if next_chunk < len(chunks):

                start_index, end_index = chunks[next_chunk]

                new_future = executor.submit(
                    search_chunk,
                    start_index,
                    end_index,
                    query_words
                )

                active_futures[new_future] = (
                    start_index,
                    end_index
                )

                next_chunk += 1

    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    all_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    execution_time = time.perf_counter() - start_time

    return all_results, execution_time


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PARALLEL SEARCH ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Number of workers
    # --------------------------------------------------------

    workers = int(
        input("Enter number of workers: ")
    )

    if workers < 1:
        print("Workers must be at least 1.")
        return

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(FILE_PATH)

    df["headline"] = df["headline"].fillna("")
    df["short_description"] = df["short_description"].fillna("")

    # --------------------------------------------------------
    # Prepare documents
    # --------------------------------------------------------

    documents = (
        df["headline"].astype(str)
        + " "
        + df["short_description"].astype(str)
    ).str.lower().tolist()

    total_documents = len(documents)

    print(f"Documents loaded: {total_documents}")

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    query = input(
        "\nEnter search query: "
    ).strip().lower()

    if not query:
        print("Search query cannot be empty.")
        return

    query_words = query.split()

    # --------------------------------------------------------
    # Create process pool
    # --------------------------------------------------------

    print("\nStarting worker processes...")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(documents,)
    ) as executor:

        # ----------------------------------------------------
        # Warm up workers
        # ----------------------------------------------------

        warmup_futures = [
            executor.submit(
                warmup_worker,
                i
            )
            for i in range(workers)
        ]

        worker_pids = set()

        for future in warmup_futures:
            worker_pids.add(
                future.result()
            )

        print(
            f"Worker processes ready: {len(worker_pids)}"
        )

        # ----------------------------------------------------
        # Perform smart parallel search
        # ----------------------------------------------------

        results, execution_time = parallel_search(
            query_words,
            total_documents,
            executor,
            workers
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    print(
        f"Matching documents: {len(results)}"
    )

    print("\nTop 10 Results:")
    print("-" * 60)

    for rank, (index, score) in enumerate(
        results[:10],
        start=1
    ):

        print(
            f"{rank}. Score: {score}"
        )

        print(
            f"   {documents[index][:200]}"
        )

        print()

    # ========================================================
    # PERFORMANCE
    # ========================================================

    print("=" * 60)
    print("PERFORMANCE")
    print("=" * 60)

    print(
        f"Workers          : {workers}"
    )

    print(
        f"Execution time   : {execution_time:.6f} seconds"
    )

    print(
        f"Results found    : {len(results)}"
    )

    # --------------------------------------------------------
    # Save execution time
    # --------------------------------------------------------

    time_file = BASE_DIR / "parallel_time.txt"

    with open(time_file, "w") as f:

        f.write(
            f"{execution_time}\n"
        )

        f.write(
            f"{workers}\n"
        )

    print(
        f"\nExecution time saved to: {time_file}"
    )


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()