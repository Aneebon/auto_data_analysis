import pandas as pd


def profile_data(df):
    profile = {
        "Total Rows": len(df),
        "Total Columns": len(df.columns),
        "Information": df.info,
        "Column Types":df.dtypes,
        "Top 5 Rows":df.head(),
        "Last 5 Rows": df.tail(),
        "Description": df.describe(),
        "Column Names": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict(),
        "Duplicate Rows": int(df.duplicated().sum()),
        

    }
    return profile
