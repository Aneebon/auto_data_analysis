
import os
import re
import itertools

import pandas as pd
import matplotlib.pyplot as plt


def safe_filename(value):
    """
    Convert a column name into a safe filename.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def save_figure(fig, filename, generated_files):
    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    generated_files.append(filename)

    print("Created:", filename)


def generate_all_charts(df, output_dir):
    """
    Generate all meaningful/applicable visualizations
    for the cleaned dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset.

    output_dir : str
        Folder where charts will be saved.

    Returns
    -------
    list
        List of generated chart file paths.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    generated_files = []

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime"]
    ).columns.tolist()


    print("\nNumeric columns:")
    print(numeric_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    print("\nDatetime columns:")
    print(datetime_columns)

    for column in numeric_columns:

        data = df[column].dropna()

        if len(data) == 0:
            continue

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.hist(
            data,
            bins=20
        )

        ax.set_title(
            f"Distribution of {column}"
        )

        ax.set_xlabel(
            column
        )

        ax.set_ylabel(
            "Frequency"
        )

        filename = os.path.join(
            output_dir,
            f"histogram_{safe_filename(column)}.png"
        )

        save_figure(
            fig,
            filename,
            generated_files
        )


    for column in categorical_columns:

        counts = df[column].value_counts(
            dropna=False
        )

        if len(counts) == 0:
            continue

        # Avoid extremely large bar charts
        if len(counts) > 20:
            counts = counts.head(20)

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            f"Count by {column}"
        )

        ax.set_xlabel(
            column
        )

        ax.set_ylabel(
            "Count"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        filename = os.path.join(
            output_dir,
            f"bar_{safe_filename(column)}.png"
        )

        save_figure(
            fig,
            filename,
            generated_files
        )


    for column in categorical_columns:

        counts = df[column].value_counts(
            dropna=False
        )

        # Pie charts only for small category counts
        if len(counts) < 2:
            continue

        if len(counts) > 8:
            continue

        fig, ax = plt.subplots(
            figsize=(7, 7)
        )

        ax.pie(
            counts.values,
            labels=counts.index.astype(str),
            autopct="%1.1f%%"
        )

        ax.set_title(
            f"Distribution of {column}"
        )

        filename = os.path.join(
            output_dir,
            f"pie_{safe_filename(column)}.png"
        )

        save_figure(
            fig,
            filename,
            generated_files
        )


    numeric_pairs = itertools.combinations(
        numeric_columns,
        2
    )

    for x, y in numeric_pairs:

        data = df[
            [x, y]
        ].dropna()

        if len(data) == 0:
            continue

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            data[x],
            data[y],
            alpha=0.5
        )

        ax.set_title(
            f"{x} vs {y}"
        )

        ax.set_xlabel(
            x
        )

        ax.set_ylabel(
            y
        )

        filename = os.path.join(
            output_dir,
            f"scatter_{safe_filename(x)}_vs_{safe_filename(y)}.png"
        )

        save_figure(
            fig,
            filename,
            generated_files
        )


    if len(numeric_columns) >= 2:

        correlation = df[
            numeric_columns
        ].corr()

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        image = ax.imshow(
            correlation,
            interpolation="nearest"
        )

        fig.colorbar(
            image,
            ax=ax
        )

        ax.set_xticks(
            range(len(numeric_columns))
        )

        ax.set_xticklabels(
            numeric_columns,
            rotation=45,
            ha="right"
        )

        ax.set_yticks(
            range(len(numeric_columns))
        )

        ax.set_yticklabels(
            numeric_columns
        )

        ax.set_title(
            "Correlation Heatmap"
        )

        filename = os.path.join(
            output_dir,
            "correlation_heatmap.png"
        )

        save_figure(
            fig,
            filename,
            generated_files
        )


    for category in categorical_columns:

        # Avoid useless high-cardinality plots
        if df[category].nunique() > 15:
            continue

        for numeric in numeric_columns:

            data = df[
                [category, numeric]
            ].dropna()

            if len(data) == 0:
                continue

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            data.boxplot(
                column=numeric,
                by=category,
                ax=ax
            )

            ax.set_title(
                f"{numeric} by {category}"
            )

            ax.set_xlabel(
                category
            )

            ax.set_ylabel(
                numeric
            )

            ax.tick_params(
                axis="x",
                rotation=45
            )

            # Remove pandas automatic subtitle
            fig.suptitle("")

            filename = os.path.join(
                output_dir,
                f"box_{safe_filename(numeric)}_by_{safe_filename(category)}.png"
            )

            save_figure(
                fig,
                filename,
                generated_files
            )


    for category in categorical_columns:

        if df[category].nunique() > 15:
            continue

        for numeric in numeric_columns:

            grouped = (
                df
                .groupby(category)[numeric]
                .mean()
                .sort_values(
                    ascending=False
                )
            )

            if len(grouped) == 0:
                continue

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            grouped.plot(
                kind="bar",
                ax=ax
            )

            ax.set_title(
                f"Average {numeric} by {category}"
            )

            ax.set_xlabel(
                category
            )

            ax.set_ylabel(
                f"Average {numeric}"
            )

            ax.tick_params(
                axis="x",
                rotation=45
            )

            filename = os.path.join(
                output_dir,
                f"mean_{safe_filename(numeric)}_by_{safe_filename(category)}.png"
            )

            save_figure(
                fig,
                filename,
                generated_files
            )


    categorical_pairs = itertools.combinations(
        categorical_columns,
        2
    )

    for x, y in categorical_pairs:

        # Avoid huge heatmaps
        if df[x].nunique() > 15:
            continue

        if df[y].nunique() > 15:
            continue

        table = pd.crosstab(
            df[x],
            df[y]
        )

        if table.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        image = ax.imshow(
            table,
            aspect="auto"
        )

        fig.colorbar(
            image,
            ax=ax
        )

        ax.set_xticks(
            range(len(table.columns))
        )

        ax.set_xticklabels(
            table.columns.astype(str),
            rotation=45,
            ha="right"
        )

        ax.set_yticks(
            range(len(table.index))
        )

        ax.set_yticklabels(
            table.index.astype(str)
        )

        ax.set_xlabel(
            y
        )

        ax.set_ylabel(
            x
        )

        ax.set_title(
            f"{x} vs {y}"
        )

        filename = os.path.join(
            output_dir,
            f"categorical_{safe_filename(x)}_vs_{safe_filename(y)}.png"
        )

        save_figure(
            fig,
            filename,
            generated_files
        )


    for date_column in datetime_columns:

        for numeric_column in numeric_columns:

            data = df[
                [date_column, numeric_column]
            ].dropna()

            if len(data) == 0:
                continue

            # Sort chronologically
            data = data.sort_values(
                by=date_column
            )

            # If there are huge numbers of timestamp observations,
            # aggregate them by day to make the chart readable.
            if data[date_column].nunique() > 1000:

                data = (
                    data
                    .set_index(date_column)
                    [numeric_column]
                    .resample("D")
                    .mean()
                    .dropna()
                    .reset_index()
                )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.plot(
                data[date_column],
                data[numeric_column]
            )

            ax.set_title(
                f"{numeric_column} over {date_column}"
            )

            ax.set_xlabel(
                date_column
            )

            ax.set_ylabel(
                numeric_column
            )

            ax.tick_params(
                axis="x",
                rotation=45
            )

            filename = os.path.join(
                output_dir,
                f"line_{safe_filename(date_column)}_vs_{safe_filename(numeric_column)}.png"
            )

            save_figure(
                fig,
                filename,
                generated_files
            )


    print(
        f"\nTotal charts generated: {len(generated_files)}"
    )

    print(
        "\nAll generated charts are stored in:"
    )

    print(
        output_dir
    )

    return generated_files