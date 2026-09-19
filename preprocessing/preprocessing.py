import pandas as pd
import numpy as np


def preprocess_data(df):

    df = df.copy()

    report = {
        "initial_rows": len(df),
        "initial_columns": len(df.columns),
        "removed_duplicate_rows": 0,
        "removed_empty_columns": [],
        "removed_empty_rows": 0,
        "cleaned_column_names": [],
        "cleaned_text_values": [],
        "converted_numeric_columns": [],
        "converted_datetime_columns": [],
        "missing_values_before": {},
        "missing_values_after": {},
        "filled_missing_numeric": {},
        "filled_missing_categorical": {},
        "infinite_values_replaced": {},
        "constant_columns": [],
        "identifier_columns": [],
        "outlier_columns": {},
        "high_missing_columns": [],
    }

    original_columns = list(df.columns)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )

    for old, new in zip(original_columns, df.columns):
        if old != new:
            report["cleaned_column_names"].append(
                f"{old} -> {new}"
            )

    before = len(df)

    df = df.dropna(how="all")

    report["removed_empty_rows"] = before - len(df)

    empty_columns = []

    for column in df.columns:

        if df[column].isna().all():
            empty_columns.append(column)

    if empty_columns:

        df = df.drop(columns=empty_columns)

        report["removed_empty_columns"] = empty_columns

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        infinite_count = np.isinf(df[column]).sum()

        if infinite_count > 0:

            df[column] = df[column].replace(
                [np.inf, -np.inf],
                np.nan
            )

            report["infinite_values_replaced"][column] = int(
                infinite_count
            )
    missing_before = df.isnull().sum()

    report["missing_values_before"] = (
        missing_before[missing_before > 0]
        .to_dict()
    )

    before = len(df)

    df = df.drop_duplicates()

    report["removed_duplicate_rows"] = (
        before - len(df)
    )


    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:

        df[column] = df[column].astype("string")

        df[column] = df[column].str.strip()

        df[column] = df[column].str.replace(
            r"\s+",
            " ",
            regex=True
        )

        df[column] = df[column].replace(
            ["", " ", "nan", "None", "null", "NULL", "N/A", "NA"],
            pd.NA
        )

        report["cleaned_text_values"].append(column)


    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:

        non_null = df[column].dropna()

        if len(non_null) == 0:
            continue

        converted = pd.to_numeric(
            non_null,
            errors="coerce"
        )

        conversion_ratio = (
            converted.notna().sum() / len(non_null)
        )
        if conversion_ratio >= 0.90:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            report["converted_numeric_columns"].append(
                column
            )



    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:

        non_null = df[column].dropna()

        if len(non_null) == 0:
            continue

        column_name = column.lower()

        if (
            "id" in column_name
            and "date" not in column_name
            and "time" not in column_name
        ):
            continue

        converted = pd.to_datetime(
            non_null,
            errors="coerce"
        )

        conversion_ratio = (
            converted.notna().sum() / len(non_null)
        )

        if conversion_ratio >= 0.90:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            report["converted_datetime_columns"].append(
                column
            )


  
    for column in df.columns:

        unique_ratio = df[column].nunique(
            dropna=True
        ) / max(len(df), 1)

        name = column.lower()

        name_based = (
            name.endswith("_id")
            or name == "id"
            or name.endswith("id")
        )

        uniqueness_based = unique_ratio >= 0.95

        if name_based or uniqueness_based:

            report["identifier_columns"].append(
                column
            )


    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        missing = df[column].isna().sum()

        if missing > 0:

            median = df[column].median()

            if not pd.isna(median):

                df[column] = df[column].fillna(
                    median
                )

                report["filled_missing_numeric"][column] = {
                    "count": int(missing),
                    "method": "median"
                }


   

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:

        missing = df[column].isna().sum()

        if missing > 0:

            mode = df[column].mode()

            if len(mode) > 0:

                df[column] = df[column].fillna(
                    mode.iloc[0]
                )

                report["filled_missing_categorical"][column] = {
                    "count": int(missing),
                    "method": "mode"
                }

            else:

                df[column] = df[column].fillna(
                    "Unknown"
                )

                report["filled_missing_categorical"][column] = {
                    "count": int(missing),
                    "method": "Unknown"
                }


 

    for column in df.columns:

        if df[column].nunique(dropna=False) <= 1:

            report["constant_columns"].append(
                column
            )


   
    for column in df.columns:

        missing_ratio = df[column].isna().mean()

        if missing_ratio >= 0.50:

            report["high_missing_columns"].append({
                "column": column,
                "missing_percentage": round(
                    missing_ratio * 100,
                    2
                )
            })



    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        # Skip identifiers
        if column in report["identifier_columns"]:
            continue

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        if IQR == 0 or pd.isna(IQR):
            continue

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        )

        count = outliers.sum()

        if count > 0:

            report["outlier_columns"][column] = {
                "count": int(count),
                "percentage": round(
                    count / len(df) * 100,
                    2
                ),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }


   

    missing_after = df.isnull().sum()

    report["missing_values_after"] = (
        missing_after[missing_after > 0]
        .to_dict()
    )


    report["final_rows"] = len(df)
    report["final_columns"] = len(df.columns)


    return df, report