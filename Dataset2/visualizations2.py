import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = (
    BASE_DIR / "News_Category_Dataset_v3.csv"
)

BENCHMARK_FILE = (
    BASE_DIR
    / "results"
    / "performance_results.csv"
)

GRAPH_DIR = (
    BASE_DIR
    / "visualizations"
)

GRAPH_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("NEWS CATEGORY DATA VISUALISATION")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

if not DATASET_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_FILE}"
    )

df = pd.read_csv(
    DATASET_FILE
)

print("Dataset loaded successfully!")


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(
    f"\nTotal articles   : {len(df):,}"
)

print(
    f"Total categories : "
    f"{df['category'].nunique()}"
)

print("\nColumns:")
print(
    df.columns.tolist()
)

print("\nFirst 5 records:")
print(
    df.head()
)


# ============================================================
# CLEAN CATEGORY DATA
# ============================================================

df["category"] = (
    df["category"]
    .fillna("UNKNOWN")
    .astype(str)
)


# ============================================================
# 1. CATEGORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("NEWS CATEGORY DISTRIBUTION")
print("=" * 70)

category_counts = (
    df["category"]
    .value_counts()
)

print(category_counts)


# ------------------------------------------------------------
# VISUALIZATION 1
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 10)
)

category_counts.sort_values().plot(
    kind="barh"
)

plt.title(
    "News Articles by Category",
    fontsize=16
)

plt.xlabel(
    "Number of Articles"
)

plt.ylabel(
    "Category"
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR
    / "01_category_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 2. TOP 10 CATEGORIES
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 NEWS CATEGORIES")
print("=" * 70)

top10_categories = (
    category_counts
    .head(10)
)

print(top10_categories)


# ------------------------------------------------------------
# VISUALIZATION 2
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 7)
)

top10_categories.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 10 News Categories",
    fontsize=16
)

plt.xlabel(
    "Number of Articles"
)

plt.ylabel(
    "Category"
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR
    / "02_top10_categories.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. ARTICLES PER MONTH
# ============================================================

print("\n" + "=" * 70)
print("ARTICLES PER MONTH")
print("=" * 70)


# Convert date column
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["month"] = (
    df["date"]
    .dt.month
)

month_order = [
    1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12
]

month_names = [
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
    df["month"]
    .value_counts()
    .reindex(month_order)
    .fillna(0)
)

print(month_counts)


# ------------------------------------------------------------
# VISUALIZATION 3
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 7)
)

plt.bar(
    month_names,
    month_counts
)

plt.title(
    "Number of Articles by Month",
    fontsize=16
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Number of Articles"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR
    / "03_articles_per_month.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. CATEGORY PERCENTAGE
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY PERCENTAGE")
print("=" * 70)

category_percentage = (
    category_counts
    / len(df)
    * 100
)

print(
    category_percentage.round(2)
)


# ------------------------------------------------------------
# VISUALIZATION 4
# ------------------------------------------------------------

top10_percentage = (
    category_percentage
    .head(10)
)

plt.figure(
    figsize=(12, 7)
)

top10_percentage.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 10 Categories by Percentage",
    fontsize=16
)

plt.xlabel(
    "Percentage of Articles (%)"
)

plt.ylabel(
    "Category"
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR
    / "04_category_percentage.png",
    dpi=300
)

plt.show()


# ============================================================
# SAVE CATEGORY SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Category": category_counts.index,
    "Article_Count": category_counts.values,
    "Percentage": [
        category_percentage[
            category
        ]
        for category in category_counts.index
    ]
})

summary_file = (
    GRAPH_DIR
    / "category_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# PARALLEL PERFORMANCE VISUALIZATION
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
    # 5. EXECUTION TIME
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

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

    plt.xlabel(
        "Number of Workers"
    )

    plt.ylabel(
        "Execution Time (seconds)"
    )

    plt.xticks(
        performance["Workers"]
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR
        / "05_execution_time_comparison.png",
        dpi=300
    )

    plt.show()


    # ========================================================
    # 6. SPEEDUP
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        performance["Workers"],
        performance["Speedup"],
        marker="o"
    )

    plt.title(
        "Parallel Search Speedup",
        fontsize=16
    )

    plt.xlabel(
        "Number of Workers"
    )

    plt.ylabel(
        "Speedup"
    )

    plt.xticks(
        performance["Workers"]
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR
        / "06_speedup.png",
        dpi=300
    )

    plt.show()


    # ========================================================
    # 7. EFFICIENCY
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        performance["Workers"],
        performance["Efficiency"],
        marker="o"
    )

    plt.title(
        "Parallel Search Efficiency",
        fontsize=16
    )

    plt.xlabel(
        "Number of Workers"
    )

    plt.ylabel(
        "Efficiency (%)"
    )

    plt.xticks(
        performance["Workers"]
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR
        / "07_efficiency.png",
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
    f"\nGraphs saved to:"
    f"\n{GRAPH_DIR}"
)

print(
    "\nGenerated visualizations:"
)

print(
    "1. Category distribution"
)

print(
    "2. Top 10 categories"
)

print(
    "3. Articles per month"
)

print(
    "4. Category percentage"
)

if BENCHMARK_FILE.exists():

    print(
        "5. Sequential vs Parallel execution time"
    )

    print(
        "6. Speedup"
    )

    print(
        "7. Efficiency"
    )

print("\nDone!")