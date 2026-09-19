import pandas as pd

from analysis.profiler import profile_data
from preprocessing.preprocessing import preprocess_data




# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(r"c:\Users\KIIT\Desktop\P\auto_data_analysis\data\Order_delivery.csv")

# ============================================================
# PROFILING
# ============================================================

profile = profile_data(df)


# ============================================================
# PREPROCESSING
# ============================================================

clean_df, preprocessing_report = preprocess_data(df)


# ============================================================
# ORIGINAL DATASET
# ============================================================

print("\n==============================")
print("ORIGINAL DATASET")
print("==============================")

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# PREPROCESSING REPORT
# ============================================================

print("\n==============================")
print("PREPROCESSING REPORT")
print("==============================")

print(
    "Duplicate rows removed:",
    preprocessing_report["removed_duplicate_rows"]
)

print(
    "Empty rows removed:",
    preprocessing_report["removed_empty_rows"]
)

print(
    "Empty columns removed:",
    preprocessing_report["removed_empty_columns"]
)

print(
    "Numeric columns converted:",
    preprocessing_report["converted_numeric_columns"]
)

print(
    "Datetime columns converted:",
    preprocessing_report["converted_datetime_columns"]
)

print(
    "Identifier columns:",
    preprocessing_report["identifier_columns"]
)

print(
    "Constant columns:",
    preprocessing_report["constant_columns"]
)

print(
    "High missing-value columns:",
    preprocessing_report["high_missing_columns"]
)

print(
    "Outliers:",
    preprocessing_report["outlier_columns"]
)


# ============================================================
# CLEANED DATASET
# ============================================================

print("\n==============================")
print("CLEANED DATASET")
print("==============================")

print(clean_df.head())

print("\nRows:", len(clean_df))
print("Columns:", len(clean_df.columns))




