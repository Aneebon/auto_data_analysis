import os
import json
import pandas as pd

from preprocessing.preprocessing import preprocess_data
from analysis.profiler import profile_data
from visualization.charts import generate_all_charts
from ai.chart_analyzer import analyze_dataset


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "movies.csv"
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)


CLEANED_FILE = os.path.join(
    OUTPUT_DIR,
    "cleaned_data.csv"
)


CHART_DIR = os.path.join(
    OUTPUT_DIR,
    "charts"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


os.makedirs(
    CHART_DIR,
    exist_ok=True
)

print("\n" + "=" * 60)
print("STEP 1 — LOADING DATASET")
print("=" * 60)


df = pd.read_csv(
    INPUT_FILE
)


print(
    "Dataset:",
    INPUT_FILE
)


print(
    "Rows:",
    len(df)
)


print(
    "Columns:",
    len(df.columns)
)


print(
    "\n Dataset loaded successfully"
)


print("\n" + "=" * 60)
print("STEP 2 — PROFILING DATASET")
print("=" * 60)


profile = profile_data(
    df
)


print(
    " Profiling completed"
)

print("\n" + "=" * 60)
print("STEP 3 — PREPROCESSING")
print("=" * 60)


clean_df, preprocessing_report = (
    preprocess_data(df)
)


print(
    "Original rows:",
    len(df)
)


print(
    "Cleaned rows:",
    len(clean_df)
)


print(
    "Duplicate rows removed:",
    preprocessing_report[
        "removed_duplicate_rows"
    ]
)


print(
    "Empty rows removed:",
    preprocessing_report[
        "removed_empty_rows"
    ]
)


print(
    "Empty columns removed:",
    preprocessing_report[
        "removed_empty_columns"
    ]
)


print(
    "Numeric columns converted:",
    preprocessing_report[
        "converted_numeric_columns"
    ]
)


print(
    "Datetime columns converted:",
    preprocessing_report[
        "converted_datetime_columns"
    ]
)


print(
    "\n✅ Preprocessing completed"
)

print("\n" + "=" * 60)
print("STEP 4 — SAVING CLEANED DATASET")
print("=" * 60)


clean_df.to_csv(
    CLEANED_FILE,
    index=False
)


print(
    "Saved:",
    CLEANED_FILE
)


print(
    " Cleaned dataset saved"
)


print("\n" + "=" * 60)
print("STEP 5 — GENERATING VISUALIZATIONS")
print("=" * 60)


generate_all_charts(
    clean_df,
    CHART_DIR
)


print(
    "\n Visualization completed"
)

print("\n" + "=" * 60)
print("STEP 6 — AI ANALYSIS")
print("=" * 60)


print(
    "\nSending cleaned dataset information"
    " to Qwen via Featherless AI..."
)


ai_response = analyze_dataset(
    clean_df
)


print(
    "\nAI RESPONSE:"
)


print(
    ai_response
)


print(
    "\n AI analysis completed"
)


print("\n" + "=" * 60)
print("STEP 7 — CHECKING AI OUTPUT")
print("=" * 60)


try:

    cleaned_response = (
        ai_response

        .replace(
            "```json",
            ""
        )

        .replace(
            "```",
            ""
        )

        .strip()
    )


    chart_plan = json.loads(
        cleaned_response
    )


    print(
        "Valid JSON: "
    )


    if "charts" in chart_plan:

        print(
            "Number of AI-selected charts:",
            len(
                chart_plan["charts"]
            )
        )


        print(
            "\nAI CHART PLAN:"
        )


        for chart in chart_plan["charts"]:

            print(
                "\n",
                chart
            )


    else:

        print(
            "Warning: charts key missing"
        )


except json.JSONDecodeError:

    print(
        "Valid JSON: "
    )
print("\n" + "=" * 60)
print("FULL SYSTEM TEST COMPLETE")
print("=" * 60)


print(
    "\nDataset:"
)


print(
    "  Loaded"
)


print(
    "Profiling:"
)


print(
    "  Completed"
)


print(
    "Preprocessing:"
)


print(
    "   Completed"
)


print(
    "Cleaned dataset:"
)


print(
    "   Saved"
)


print(
    "Visualization:"
)


print(
    "  Generated"
)


print(
    "AI:"
)


print(
    "  Qwen + Featherless responded"
)


print(
    "\nCharts location:"
)


print(
    CHART_DIR
)


print(
    "\n COMPLETE SYSTEM PIPELINE FINISHED!"
)