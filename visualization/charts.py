import os
import itertools

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# GENERATE ALL CHARTS
# ============================================================

def generate_all_charts(df, output_dir):
    """
    Generate automatic visualizations for the cleaned dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset.

    output_dir : str
        Folder where charts will be saved.
    """

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("\nDataset received by visualization engine!")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))


    # ========================================================
    # DETECT COLUMN TYPES
    # ========================================================

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()


    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()


    datetime_columns = df.select_dtypes(
        include=["datetime"]
    ).columns.tolist()


    print("\n==============================")
    print("COLUMN TYPES")
    print("==============================")


    print("\nNumeric columns:")
    print(numeric_columns)


    print("\nCategorical columns:")
    print(categorical_columns)


    print("\nDatetime columns:")
    print(datetime_columns)


    # ========================================================
    # 1. HISTOGRAMS
    # ========================================================

    print("\n==============================")
    print("GENERATING HISTOGRAMS")
    print("==============================")


    for column in numeric_columns:

        data = df[column].dropna()

        if len(data) == 0:
            continue


        plt.figure(
            figsize=(8, 5)
        )


        plt.hist(
            data,
            bins=20
        )


        plt.title(
            f"Distribution of {column}"
        )


        plt.xlabel(
            column
        )


        plt.ylabel(
            "Frequency"
        )


        plt.tight_layout()


        filename = os.path.join(
            output_dir,
            f"histogram_{column}.png"
        )


        plt.savefig(
            filename,
            dpi=150,
            bbox_inches="tight"
        )


        plt.close()


        print(
            "Created:",
            filename
        )


    # ========================================================
    # 2. BAR CHARTS
    # ========================================================

    print("\n==============================")
    print("GENERATING BAR CHARTS")
    print("==============================")


    for column in categorical_columns:

        counts = df[column].value_counts(
            dropna=False
        )


        # Avoid useless high-cardinality graphs
        if len(counts) == 0:
            continue


        # If there are too many categories,
        # display only top 20.
        if len(counts) > 20:

            counts = counts.head(20)


        plt.figure(
            figsize=(10, 6)
        )


        counts.plot(
            kind="bar"
        )


        plt.title(
            f"Count by {column}"
        )


        plt.xlabel(
            column
        )


        plt.ylabel(
            "Count"
        )


        plt.xticks(
            rotation=45,
            ha="right"
        )


        plt.tight_layout()


        filename = os.path.join(
            output_dir,
            f"bar_{column}.png"
        )


        plt.savefig(
            filename,
            dpi=150,
            bbox_inches="tight"
        )


        plt.close()


        print(
            "Created:",
            filename
        )


    # ========================================================
    # 3. PIE CHARTS
    # ========================================================

    print("\n==============================")
    print("GENERATING PIE CHARTS")
    print("==============================")


    for column in categorical_columns:

        counts = df[column].value_counts(
            dropna=False
        )


        # Pie charts only make sense
        # for a small number of categories.

        if len(counts) < 2:
            continue


        if len(counts) > 8:
            continue


        plt.figure(
            figsize=(7, 7)
        )


        plt.pie(
            counts.values,
            labels=counts.index.astype(str),
            autopct="%1.1f%%"
        )


        plt.title(
            f"Distribution of {column}"
        )


        filename = os.path.join(
            output_dir,
            f"pie_{column}.png"
        )


        plt.savefig(
            filename,
            dpi=150,
            bbox_inches="tight"
        )


        plt.close()


        print(
            "Created:",
            filename
        )


    # ========================================================
    # 4. SCATTER PLOTS
    #
    # EVERY NUMERIC COLUMN AGAINST EVERY OTHER
    # ========================================================

    print("\n==============================")
    print("GENERATING SCATTER PLOTS")
    print("==============================")


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


        plt.figure(
            figsize=(8, 5)
        )


        plt.scatter(
            data[x],
            data[y],
            alpha=0.5
        )


        plt.title(
            f"{x} vs {y}"
        )


        plt.xlabel(
            x
        )


        plt.ylabel(
            y
        )


        plt.tight_layout()


        filename = os.path.join(
            output_dir,
            f"scatter_{x}_vs_{y}.png"
        )


        plt.savefig(
            filename,
            dpi=150,
            bbox_inches="tight"
        )


        plt.close()


        print(
            "Created:",
            filename
        )


    # ========================================================
    # 5. CORRELATION HEATMAP
    # ========================================================

    print("\n==============================")
    print("GENERATING CORRELATION HEATMAP")
    print("==============================")


    if len(numeric_columns) >= 2:

        correlation = df[
            numeric_columns
        ].corr()


        plt.figure(
            figsize=(10, 8)
        )


        plt.imshow(
            correlation,
            interpolation="nearest"
        )


        plt.colorbar()


        plt.xticks(
            range(len(numeric_columns)),
            numeric_columns,
            rotation=45,
            ha="right"
        )


        plt.yticks(
            range(len(numeric_columns)),
            numeric_columns
        )


        plt.title(
            "Correlation Heatmap"
        )


        plt.tight_layout()


        filename = os.path.join(
            output_dir,
            "correlation_heatmap.png"
        )


        plt.savefig(
            filename,
            dpi=150,
            bbox_inches="tight"
        )


        plt.close()


        print(
            "Created:",
            filename
        )


    # ========================================================
    # 6. BOX PLOTS
    #
    # CATEGORICAL vs NUMERIC
    # ========================================================

    print("\n==============================")
    print("GENERATING BOX PLOTS")
    print("==============================")


    for category in categorical_columns:

        # Skip high-cardinality columns
        if df[category].nunique() > 15:
            continue


        for numeric in numeric_columns:

            data = df[
                [category, numeric]
            ].dropna()


            if len(data) == 0:
                continue


            plt.figure(
                figsize=(10, 6)
            )


            data.boxplot(
                column=numeric,
                by=category
            )


            plt.title(
                f"{numeric} by {category}"
            )


            plt.suptitle("")


            plt.xlabel(
                category
            )


            plt.ylabel(
                numeric
            )


            plt.xticks(
                rotation=45,
                ha="right"
            )


            plt.tight_layout()


            filename = os.path.join(
                output_dir,
                f"box_{numeric}_by_{category}.png"
            )


            plt.savefig(
                filename,
                dpi=150,
                bbox_inches="tight"
            )


            plt.close()


            print(
                "Created:",
                filename
            )


    # ========================================================
    # 7. MEAN BAR CHARTS
    #
    # CATEGORICAL vs NUMERIC
    # ========================================================

    print("\n==============================")
    print("GENERATING GROUPED MEAN CHARTS")
    print("==============================")


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


            plt.figure(
                figsize=(10, 6)
            )


            grouped.plot(
                kind="bar"
            )


            plt.title(
                f"Average {numeric} by {category}"
            )


            plt.xlabel(
                category
            )


            plt.ylabel(
                f"Average {numeric}"
            )


            plt.xticks(
                rotation=45,
                ha="right"
            )


            plt.tight_layout()


            filename = os.path.join(
                output_dir,
                f"mean_{numeric}_by_{category}.png"
            )


            plt.savefig(
                filename,
                dpi=150,
                bbox_inches="tight"
            )


            plt.close()


            print(
                "Created:",
                filename
            )


    # ========================================================
    # 8. CATEGORICAL vs CATEGORICAL
    #
    # CROSS-TAB HEATMAP
    # ========================================================

    print("\n==============================")
    print("GENERATING CATEGORICAL HEATMAPS")
    print("==============================")


    categorical_pairs = itertools.combinations(
        categorical_columns,
        2
    )


    for x, y in categorical_pairs:

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


        plt.figure(
            figsize=(10, 7)
        )


        plt.imshow(
            table,
            aspect="auto"
        )


        plt.colorbar()


        plt.xticks(
            range(len(table.columns)),
            table.columns.astype(str),
            rotation=45,
            ha="right"
        )


        plt.yticks(
            range(len(table.index)),
            table.index.astype(str)
        )


        plt.xlabel(
            y
        )


        plt.ylabel(
            x
        )


        plt.title(
            f"{x} vs {y}"
        )


        plt.tight_layout()


        filename = os.path.join(
            output_dir,
            f"categorical_{x}_vs_{y}.png"
        )


        plt.savefig(
            filename,
            dpi=150,
            bbox_inches="tight"
        )


        plt.close()


        print(
            "Created:",
            filename
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n==============================")
    print("CHART GENERATION COMPLETE")
    print("==============================")


    print(
        "\nAll generated charts are stored in:"
    )


    print(
        output_dir
    )