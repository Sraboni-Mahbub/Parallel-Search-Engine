
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "News_Category_Dataset_v3.csv"

GRAPH_DIR = BASE_DIR / "visualizations"
GRAPH_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

if not DATASET_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded successfully!")


# ============================================================
# CLEAN DATA
# ============================================================

df["category"] = df["category"].fillna("UNKNOWN").astype(str)
df["headline"] = df["headline"].fillna("").astype(str)

# Calculate headline length
df["headline_length"] = df["headline"].str.len()


# ============================================================
# GRAPH 1: NEWS CATEGORY DISTRIBUTION
# ============================================================

category_counts = df["category"].value_counts()

plt.figure(figsize=(14, 10))

category_counts.sort_values().plot(kind="barh")

plt.title("News Articles by Category")
plt.xlabel("Number of Articles")
plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "01_category_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 2: TOP 10 NEWS CATEGORIES
# ============================================================

top10_categories = category_counts.head(10)

plt.figure(figsize=(12, 7))

top10_categories.sort_values().plot(kind="barh")

plt.title("Top 10 News Categories")
plt.xlabel("Number of Articles")
plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "02_top10_categories.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 3: AVERAGE HEADLINE LENGTH BY CATEGORY
# ============================================================

avg_headline_length = (
    df.groupby("category")["headline_length"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(14, 10))

avg_headline_length.plot(kind="barh")

plt.title("Average Headline Length by Category")
plt.xlabel("Average Headline Length (Characters)")
plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "03_avg_headline_by_category.png",
    dpi=300
)

plt.show()


# ============================================================
# FINISHED
# ============================================================

print("\nVisualization completed!")
print(f"Graphs saved in: {GRAPH_DIR}")

print("\nGenerated graphs:")
print("1. News Category Distribution")
print("2. Top 10 News Categories")
print("3. Average Headline Length by Category")

