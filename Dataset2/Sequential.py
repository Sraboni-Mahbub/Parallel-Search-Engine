import pandas as pd
import time
import re


# =========================
# LOAD DATASET
# =========================

file_path = "Dataset2/News_Category_Dataset_v3.csv"

df = pd.read_csv(file_path)


# Replace missing values
df["headline"] = df["headline"].fillna("")
df["short_description"] = df["short_description"].fillna("")


print("Total documents:", len(df))


# =========================
# PREPARE DOCUMENTS
# NOT TIMED
# =========================

documents = (
    df["headline"].astype(str)
    + " "
    + df["short_description"].astype(str)
).tolist()


# =========================
# USER INPUT
# NOT TIMED
# =========================

query = input("\nEnter search query: ")


# Process query before timer
query_words = query.lower().split()


# =========================
# SEQUENTIAL SEARCH
# =========================

def sequential_search(query_words, documents):

    results = []


    # ==================================
    # TIMER STARTS HERE
    # ==================================

    start_time = time.perf_counter()


    # Search documents one by one
    for index, document in enumerate(documents):

        text = document.lower()

        score = 0


        # Search each query word
        for word in query_words:

            occurrences = len(
                re.findall(
                    r"\b" + re.escape(word) + r"\b",
                    text
                )
            )

            score += occurrences


        # Store matching document
        if score > 0:
            results.append((index, score))


    # Sort results from highest score to lowest
    results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # ==================================
    # TIMER ENDS HERE
    # ==================================

    end_time = time.perf_counter()

    execution_time = end_time - start_time


    return results, execution_time


# =========================
# RUN SEARCH
# =========================

results, execution_time = sequential_search(
    query_words,
    documents
)


# =========================
# DISPLAY RESULTS
# NOT TIMED
# =========================

print("\nSEARCH RESULTS")

print("Query:", query)

print("Documents searched:", len(documents))

print("Matching documents:", len(results))


print("\nTop 10 Results:")


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


print("\nTIME CALCULATION")

print(
    f"Sequential execution time: "
    f"{execution_time:.6f} seconds"
)