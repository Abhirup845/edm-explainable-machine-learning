import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Load the EDM dataset
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "EDM_dataset_262_samples.csv"
df = pd.read_csv(CSV_PATH)

# Basic information
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nBasic statistics:")
print(df.describe())


import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Distribution of Surface Roughness
# -------------------------------

plt.figure(figsize=(7, 5))
plt.hist(df["SR_um"], bins=20)
plt.xlabel("Surface Roughness (µm)")
plt.ylabel("Number of Experiments")
plt.title("Distribution of Surface Roughness")
plt.show()


# -------------------------------
# 2. Correlation Matrix
# -------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.show()