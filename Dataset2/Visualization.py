import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = BASE_DIR / "News_Category_Dataset_v3.csv"

# Optional benchmark file
BENCHMARK_FILE = BASE_DIR / "results" / "performance_results.csv"

# Folder where graphs will be saved
GRAPH_DIR = BASE_DIR / "visualizations"
GRAPH_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("NEWS CATEGORY DATA VISUALIZATION")
print("=" * 70)

print("\nLoading dataset...")

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_FILE}"
    )

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded successfully!")


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"\nTotal articles   : {len(df):,}")
print(f"Total categories : {df['category'].nunique()}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# DATA CLEANING
# ============================================================

df["headline"] = df["headline"].fillna("").astype(str)
df["short_description"] = (
    df["short_description"]
    .fillna("")
    .astype(str)
)

df["authors"] = df["authors"].fillna("").astype(str)

df["category"] = (
    df["category"]
    .fillna("UNKNOWN")
    .astype(str)
)


# ============================================================
# DATE PROCESSING
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.strftime("%B")


# ============================================================
# TEXT LENGTH
# ============================================================

df["headline_length"] = df["headline"].str.len()

df["description_length"] = (
    df["short_description"].str.len()
)


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

category_counts = (
    df["category"]
    .value_counts()
)

print("\n" + "=" * 70)
print("NEWS CATEGORY DISTRIBUTION")
print("=" * 70)

print(category_counts)


# ============================================================
# TOP 10 CATEGORIES
# ============================================================

top10_categories = category_counts.head(10)

print("\nTop 10 categories:")
print(top10_categories)


# ============================================================
# FIGURE 1 - ALL CATEGORY DISTRIBUTION
# ============================================================

plt.figure(figsize=(14, 10))

category_counts.sort_values().plot(
    kind="barh"
)

plt.title(
    "News Articles by Category",
    fontsize=16
)

plt.xlabel("Number of Articles")
plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "01_category_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# FIGURE 2 - TOP 10 CATEGORIES
# ============================================================

plt.figure(figsize=(12, 7))

top10_categories.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 10 News Categories",
    fontsize=16
)

plt.xlabel("Number of Articles")
plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "02_top10_categories.png",
    dpi=300
)

plt.show()


# ============================================================
# HEADLINE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("HEADLINE LENGTH")
print("=" * 70)

print(
    f"Average headline length: "
    f"{df['headline_length'].mean():.2f} characters"
)

print(
    f"Maximum headline length: "
    f"{df['headline_length'].max()} characters"
)

print(
    f"Minimum headline length: "
    f"{df['headline_length'].min()} characters"
)


# ============================================================
# FIGURE 3 - HEADLINE LENGTH
# ============================================================

plt.figure(figsize=(12, 7))

plt.hist(
    df["headline_length"],
    bins=50
)

plt.title(
    "Distribution of Headline Length",
    fontsize=16
)

plt.xlabel("Headline Length (characters)")
plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "03_headline_length.png",
    dpi=300
)

plt.show()


# ============================================================
# DESCRIPTION STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("DESCRIPTION LENGTH")
print("=" * 70)

print(
    f"Average description length: "
    f"{df['description_length'].mean():.2f} characters"
)

print(
    f"Maximum description length: "
    f"{df['description_length'].max()} characters"
)

print(
    f"Minimum description length: "
    f"{df['description_length'].min()} characters"
)


# ============================================================
# FIGURE 4 - DESCRIPTION LENGTH
# ============================================================

plt.figure(figsize=(12, 7))

plt.hist(
    df["description_length"],
    bins=50
)

plt.title(
    "Distribution of Short Description Length",
    fontsize=16
)

plt.xlabel(
    "Description Length (characters)"
)

plt.ylabel(
    "Number of Articles"
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "04_description_length.png",
    dpi=300
)

plt.show()


# ============================================================
# AVERAGE HEADLINE LENGTH BY CATEGORY
# ============================================================

avg_headline_length = (
    df.groupby("category")["headline_length"]
    .mean()
    .sort_values()
)

print("\n" + "=" * 70)
print("AVERAGE HEADLINE LENGTH BY CATEGORY")
print("=" * 70)

print(avg_headline_length)


# ============================================================
# FIGURE 5 - AVERAGE HEADLINE LENGTH BY CATEGORY
# ============================================================

plt.figure(figsize=(14, 10))

avg_headline_length.plot(
    kind="barh"
)

plt.title(
    "Average Headline Length by Category",
    fontsize=16
)

plt.xlabel(
    "Average Headline Length (characters)"
)

plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "05_avg_headline_by_category.png",
    dpi=300
)

plt.show()


# ============================================================
# ARTICLES PER YEAR
# ============================================================

year_counts = (
    df["year"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 70)
print("ARTICLES PER YEAR")
print("=" * 70)

print(year_counts)


# ============================================================
# FIGURE 6 - ARTICLES PER YEAR
# ============================================================

plt.figure(figsize=(12, 7))

year_counts.plot(
    kind="bar"
)

plt.title(
    "Number of Articles by Year",
    fontsize=16
)

plt.xlabel("Year")
plt.ylabel("Number of Articles")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "06_articles_per_year.png",
    dpi=300
)

plt.show()


# ============================================================
# ARTICLES PER MONTH
# ============================================================

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

month_counts = (
    df["month_name"]
    .value_counts()
    .reindex(month_order)
)


# ============================================================
# FIGURE 7 - ARTICLES PER MONTH
# ============================================================

plt.figure(figsize=(13, 7))

month_counts.plot(
    kind="bar"
)

plt.title(
    "Number of Articles by Month",
    fontsize=16
)

plt.xlabel("Month")
plt.ylabel("Number of Articles")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "07_articles_per_month.png",
    dpi=300
)

plt.show()


# ============================================================
# MISSING VALUES
# ============================================================

missing_values = (
    df[
        [
            "headline",
            "category",
            "short_description",
            "authors",
            "date"
        ]
    ]
    .isnull()
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(missing_values)


# ============================================================
# FIGURE 8 - MISSING VALUES
# ============================================================

plt.figure(figsize=(10, 6))

missing_values.plot(
    kind="bar"
)

plt.title(
    "Missing Values by Column",
    fontsize=16
)

plt.xlabel("Column")
plt.ylabel("Number of Missing Values")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "08_missing_values.png",
    dpi=300
)

plt.show()


# ============================================================
# CATEGORY PERCENTAGE
# ============================================================

category_percentage = (
    category_counts
    / len(df)
    * 100
)

print("\n" + "=" * 70)
print("CATEGORY PERCENTAGE")
print("=" * 70)

print(
    category_percentage.round(2)
)


# ============================================================
# FIGURE 9 - CATEGORY PERCENTAGE TOP 10
# ============================================================

top10_percentage = category_percentage.head(10)

plt.figure(figsize=(12, 7))

top10_percentage.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 10 Categories by Percentage",
    fontsize=16
)

plt.xlabel("Percentage of Articles (%)")
plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "09_category_percentage.png",
    dpi=300
)

plt.show()


# ============================================================
# SAVE DATASET SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Category": category_counts.index,
    "Article_Count": category_counts.values,
    "Percentage": [
        category_percentage[c]
        for c in category_counts.index
    ]
})

summary_file = (
    GRAPH_DIR / "category_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# PARALLEL SEARCH BENCHMARK
# ============================================================

print("\n" + "=" * 70)
print("PARALLEL SEARCH PERFORMANCE")
print("=" * 70)


if BENCHMARK_FILE.exists():

    performance = pd.read_csv(
        BENCHMARK_FILE
    )

    print("\nPerformance data:")
    print(performance)


    # ========================================================
    # FIGURE 10 - SEQUENTIAL VS PARALLEL
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        performance["Workers"],
        performance["Sequential_Time"],
        marker="o",
        label="Sequential"
    )

    plt.plot(
        performance["Workers"],
        performance["Parallel_Time"],
        marker="o",
        label="Parallel"
    )

    plt.title(
        "Sequential vs Parallel Execution Time",
        fontsize=16
    )

    plt.xlabel("Number of Workers")
    plt.ylabel("Execution Time (seconds)")

    plt.xticks(
        performance["Workers"]
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "10_execution_time_comparison.png",
        dpi=300
    )

    plt.show()


    # ========================================================
    # FIGURE 11 - SPEEDUP
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        performance["Workers"],
        performance["Speedup"],
        marker="o"
    )

    plt.title(
        "Parallel Search Speedup",
        fontsize=16
    )

    plt.xlabel("Number of Workers")
    plt.ylabel("Speedup")

    plt.xticks(
        performance["Workers"]
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "11_speedup.png",
        dpi=300
    )

    plt.show()


    # ========================================================
    # FIGURE 12 - EFFICIENCY
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        performance["Workers"],
        performance["Efficiency"],
        marker="o"
    )

    plt.title(
        "Parallel Search Efficiency",
        fontsize=16
    )

    plt.xlabel("Number of Workers")
    plt.ylabel("Efficiency (%)")

    plt.xticks(
        performance["Workers"]
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "12_efficiency.png",
        dpi=300
    )

    plt.show()


else:

    print(
        "\nNo performance_results.csv found."
    )

    print(
        "Dataset visualizations will still be generated."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)

print(
    f"\nGraphs saved to:\n{GRAPH_DIR}"
)

print(
    f"\nCategory summary saved to:\n{summary_file}"
)

print(
    "\nGenerated visualizations:"
)

print("1. Category distribution")
print("2. Top 10 categories")
print("3. Headline length")
print("4. Description length")
print("5. Average headline length by category")
print("6. Articles per year")
print("7. Articles per month")
print("8. Missing values")
print("9. Category percentage")

if BENCHMARK_FILE.exists():
    print("10. Sequential vs Parallel execution time")
    print("11. Speedup")
    print("12. Efficiency")

print("\nDone!")