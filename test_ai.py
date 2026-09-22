import pandas as pd

from auto_data_analysis.ai.chart_analyzer import analyze_dataset
df = pd.read_csv(
    "output/cleaned_data.csv"
)
print("\nSending dataset information to Hugging Face...")
response = analyze_dataset(df)
print(response)
