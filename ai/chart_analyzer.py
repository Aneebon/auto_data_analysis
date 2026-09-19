import os
import json
import pandas as pd

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv(
    "HUGGINGFACEHUB_API_TOKEN"
)
if not HF_TOKEN:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN not found in .env"
    )
client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN
)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def create_dataset_metadata(df):

    metadata = {
        "rows": len(df),
        "columns": []
    }

    for column in df.columns:

        column_info = {
            "name": column,
            "dtype": str(df[column].dtype),
            "unique_values": int(
                df[column].nunique()
            ),
            "missing_values": int(
                df[column].isna().sum()
            )
        }
        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            column_info["min"] = float(
                df[column].min()
            )

            column_info["max"] = float(
                df[column].max()
            )

            column_info["mean"] = float(
                df[column].mean()
            )

        elif df[column].dtype == "object":

            column_info["sample_values"] = (
                df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()[:10]
            )

        metadata["columns"].append(
            column_info
        )

    return metadata
def analyze_dataset(df):

    metadata = create_dataset_metadata(df)
    metadata_json = json.dumps(
        metadata,
        indent=2
    )
    system_prompt = """
You are an expert data analysis and
data visualization assistant.

Analyze the provided dataset metadata.

Your task is to recommend useful visualizations.

Available chart types:

- histogram
- bar
- pie
- scatter
- line

Rules:

1. Use histogram for numeric distributions.

2. Use bar charts for categorical comparisons.

3. Use pie charts only when a categorical
   column has a small number of categories.

4. Use scatter plots for relationships
   between two numeric columns.

5. Use line charts when a date/time column
   and numeric column indicate a meaningful trend.

6. Never use identifier columns.

7. Never use columns where almost every
   value is unique as categorical charts.

8. Only use columns that actually exist.

9. Select the most useful visualizations.

10. Return between 3 and 8 charts.

11. Return ONLY valid JSON.

The JSON must contain a "charts" array.

For histogram:

type, column, title

For bar:

type, column, title

For pie:

type, column, title

For scatter:

type, x, y, title

For line:

type, x, y, title
"""

    user_prompt = f"""
Here is the dataset metadata:

{metadata_json}

Analyze this dataset and return the
visualization plan as JSON.
"""
    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_tokens=1000,

        temperature=0.1
    )
    return response.choices[0].message.content