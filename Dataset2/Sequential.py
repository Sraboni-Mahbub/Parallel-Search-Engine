import pandas as pd
import time
import re
from pathlib import Path


# ============================================================
# FILE PATH
# ============================================================

BASE_DIR = Path(__file__).parent

FILE_PATH = (
    BASE_DIR
    / "News_Category_Dataset_v3.csv"
)


# ============================================================
# SEQUENTIAL SEARCH FUNCTION
# ============================================================

def sequential_search(query_words, documents):

    results = []

    # Compile patterns before timing
    patterns = [
        re.compile(
            r"\b" + re.escape(word) + r"\b"
        )
        for word in query_words
    ]

    # ========================================================
    # TIMER START
    # ========================================================

    start_time = time.perf_counter()


    # ========================================================
    # SEARCH DOCUMENTS
    # ========================================================

    for index, text in enumerate(documents):

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
        end_time - start_time
    )


    return results, execution_time


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # ========================================================
    # LOAD DATASET
    # ========================================================

    if not FILE_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{FILE_PATH}"
        )


    df = pd.read_csv(
        FILE_PATH
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


    # ========================================================
    # PREPARE DOCUMENTS
    # NOT TIMED
    # ========================================================

    documents = (

        df["headline"].astype(str)

        + " "

        + df["short_description"].astype(str)

    ).str.lower().tolist()


    # ========================================================
    # USER INPUT
    # NOT TIMED
    # ========================================================

    query = input(
        "\nEnter search query: "
    ).strip()


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
    # RUN SEQUENTIAL SEARCH
    # ========================================================

    results, execution_time = sequential_search(
        query_words,
        documents
    )


    # ========================================================
    # DISPLAY RESULTS
    # NOT TIMED
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "SEQUENTIAL SEARCH RESULTS"
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


    # ========================================================
    # TOP 10 RESULTS
    # ========================================================

    print(
        "\nTop 10 Results:"
    )


    if not results:

        print(
            "No matching documents found."
        )


    else:

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
        "SEQUENTIAL SEARCH PERFORMANCE"
    )

    print(
        "========================================"
    )


    print(
        f"Execution time: "
        f"{execution_time:.6f} seconds"
    )


    # ========================================================
    # SAVE TIME FOR BENCHMARK
    # ========================================================

    output_file = (
        BASE_DIR
        / "sequential_time.txt"
    )


    with open(
        output_file,
        "w"
    ) as file:

        file.write(
            str(execution_time)
        )


    print(
        f"\nExecution time saved to:\n"
        f"{output_file}"
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()