from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent
input_file = folder / "crop_yield_messy.csv"
output_file = folder / "crop_yield_cleaned.csv"

if not input_file.exists():
    raise SystemExit(
        "Cannot find crop_yield_messy.csv. "
        "Put it in the same folder as messycleandemo.py."
    )

# Load the messy CSV
df = pd.read_csv(input_file)
print("Rows loaded:", len(df))
print("\nFirst 5 rows:")
print(df.head())

# Remove the unnecessary column that was added for the demo
if "Temporary_Notes" in df.columns:
    df = df.drop(columns=["Temporary_Notes"])
    print("\nRemoved the demo-only Temporary_Notes column.")

# Trim extra spaces and standardize spaces inside category labels
for column in ["Crop", "Season", "State"]:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

# Replace the Gender mapping with a mapping for the six seasons in your data
season_map = {
    "KHARIF": "Kharif",
    "RABI": "Rabi",
    "WHOLE YEAR": "Whole Year",
    "SUMMER": "Summer",
    "AUTUMN": "Autumn",
    "WINTER": "Winter",
}

season_clean = df["Season"].str.upper()
df["Season"] = season_clean.map(season_map).fillna(df["Season"])

# Build a crop-name mapping from crop names actually found in this CSV.
# This standardizes case/spacing without inventing new crop categories.
crop_names = df["Crop"].dropna().unique()
crop_map = {}

for crop in crop_names:
    crop_map.setdefault(crop.casefold(), crop)

crop_key = df["Crop"].str.casefold()
df["Crop"] = crop_key.map(crop_map).fillna(df["Crop"])

print("\nCrop names recognized from the CSV:")
print(", ".join(sorted(df["Crop"].dropna().unique())))

# Convert numeric columns; invalid text becomes missing for review/removal below
numeric_columns = [
    "Crop_Year",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove exact duplicate rows
duplicate_count = int(df.duplicated().sum())
df = df.drop_duplicates()

# For this deliberately messy practice file, remove rows with missing or
# invalid values in the fields needed for the project.
required_columns = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield",
]

rows_before_missing_check = len(df)
print("\nMissing values before removing affected rows:")
print(df[required_columns].isna().sum())

df = df.dropna(subset=required_columns)
missing_rows_removed = rows_before_missing_check - len(df)

# Save a separate cleaned copy. The messy input is left unchanged.
df.to_csv(output_file, index=False)

print("\nDuplicate rows removed:", duplicate_count)
print("Rows removed for missing/invalid required values:", missing_rows_removed)
print("Rows in cleaned file:", len(df))
print("Cleaned file saved as:", output_file.name)