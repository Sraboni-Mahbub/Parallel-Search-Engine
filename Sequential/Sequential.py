import pandas as pd
import time
import re
from pathlib import Path


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Find the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset location
file_path = BASE_DIR / "data" / "NewsCategorizer.csv"

# Load dataset
df = pd.read_csv(file_path)

# Replace missing values with empty strings
df["headline"] = df["headline"].fillna("")
df["short_description"] = df["short_description"].fillna("")
df["keywords"] = df["keywords"].fillna("")

print("Total documents:", len(df))


# ============================================================
# 2. PREPARE SEARCHABLE DOCUMENTS
# ============================================================

documents = (
    df["headline"].astype(str)
    + " "
    + df["short_description"].astype(str)
    + " "
    + df["keywords"].astype(str)
).tolist()


# ============================================================
# 3. SEQUENTIAL SEARCH FUNCTION
# ============================================================

def sequential_search(query, documents):

    # Convert query into individual words
    query_words = query.lower().split()

    results = []

    # --------------------------------------------------------
    # Start timing ONLY the search
    # --------------------------------------------------------

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Search every document ONE BY ONE
    # --------------------------------------------------------

    for index, document in enumerate(documents):

        text = document.lower()

        score = 0

        # Search every query word
        for word in query_words:

            occurrences = len(
                re.findall(
                    r"\b" + re.escape(word) + r"\b",
                    text
                )
            )

            score += occurrences

        # Save matching documents
        if score > 0:

            results.append(
                (index, score)
            )

    # --------------------------------------------------------
    # Rank results by relevance
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # --------------------------------------------------------
    # Stop timing
    # --------------------------------------------------------

    end_time = time.perf_counter()

    execution_time = end_time - start_time

    return results, execution_time


# ============================================================
# 4. GET USER QUERY
# ============================================================

query = input("\nEnter search query: ")


# ============================================================
# 5. PERFORM SEQUENTIAL SEARCH
# ============================================================

results, execution_time = sequential_search(
    query,
    documents
)


# ============================================================
# 6. DISPLAY SEARCH RESULTS
# ============================================================

print("\n" + "=" * 70)
print("              SEQUENTIAL SEARCH ENGINE")
print("=" * 70)

print("Query:", query)

print("Documents searched:", len(documents))

print("Matching documents:", len(results))

print("\nTop 10 Results")
print("-" * 70)


for rank, (index, score) in enumerate(
    results[:10],
    start=1
):

    print(f"\n{rank}. {df.iloc[index]['headline']}")

    print(
        "Category:",
        df.iloc[index]["category"]
        if "category" in df.columns
        else "N/A"
    )

    print(
        "Score:",
        score
    )

    print(
        "Description:",
        df.iloc[index]["short_description"]
    )


# ============================================================
# 7. PERFORMANCE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("           SEQUENTIAL SEARCH PERFORMANCE")
print("=" * 70)

print(
    f"Execution time: "
    f"{execution_time:.6f} seconds"
)

print("=" * 70)