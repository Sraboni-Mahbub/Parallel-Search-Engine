import pandas as pd
import time
import re


#Load the dataset

file_path = "NewsCategorizer.csv"

df = pd.read_csv(file_path)

# Replace missing values with empty strings
df["headline"] = df["headline"].fillna("")
df["short_description"] = df["short_description"].fillna("")
df["keywords"] = df["keywords"].fillna("")

print("Total documents:", len(df))


#Seach documents

documents = (
    df["headline"].astype(str)
    + " "
    + df["short_description"].astype(str)
    + " "
    + df["keywords"].astype(str)
).tolist()

#User enters search query

query = input("\nEnter search query: ")

#Search Function for sequential

def sequential_search(query, documents):

    query_words = query.lower().split()

    results = []

    # Start timing only the search
    start_time = time.perf_counter()

    # Search every document ONE BY ONE
    for index, document in enumerate(documents):

        text = document.lower()

        score = 0

        # Count occurrences of every query word
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
            results.append((index, score))

    # Rank results by relevance
    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Stop timer
    end_time = time.perf_counter()

    execution_time = end_time - start_time

    return results, execution_time





#Perform sequential search

results, execution_time = sequential_search(
    query,
    documents
)


#results

print("\n" + "=" * 60)

print("SEQUENTIAL SEARCH ENGINE")

print("=" * 60)

print("Query:", query)

print("Documents searched:", len(documents))

print("Matching documents:", len(results))

print("\nTop 10 Results")

print("-" * 60)


for rank, (index, score) in enumerate(
    results[:10],
    start=1
):

    print(f"\n{rank}. {df.iloc[index]['headline']}")

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


#Display execution time

print("\n" + "=" * 60)

print("SEQUENTIAL SEARCH PERFORMANCE")

print("=" * 60)

print(
    f"Execution time: "
    f"{execution_time:.6f} seconds"
)