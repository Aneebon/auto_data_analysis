import pandas as pd

from auto_data_analysis.ai.chart_analyzer import analyze_dataset


# ============================================================
# LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv(
    "output/cleaned_data.csv"
)


# ============================================================
# ASK AI
# ============================================================

print("\nSending dataset information to Hugging Face...")

response = analyze_dataset(df)


# ============================================================
# DISPLAY RESPONSE
# ============================================================

print("\n==============================")
print("AI CHART PLAN")
print("==============================")

print(response)