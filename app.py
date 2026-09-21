import streamlit as st
import pandas as pd
import os

from analysis.profiler import profile_data
from preprocessing.preprocessing import preprocess_data
from visualization.charts import generate_all_charts

st.set_page_config(
    page_title="AI Data Analytics",
    page_icon="",
    layout="wide"
)

st.title("AI Data Analytics")

st.write(
    "Upload a dataset and let the system "
    "profile, preprocess and automatically visualize it."
)

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Upload a CSV file to begin."
    )

    st.stop()

try:

    df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Error loading dataset: {e}"
    )

    st.stop()


st.success(
    "Dataset loaded successfully! "
)

st.header(
    "Dataset Overview"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Rows",
        len(df)
    )


with col2:

    st.metric(
        "Columns",
        len(df.columns)
    )


with col3:

    st.metric(
        "Missing Values",
        int(
            df.isna().sum().sum()
        )
    )

st.subheader(
    "Dataset Preview"
)


st.dataframe(
    df.head(10),
    use_container_width=True
)

st.header(
    "Data Profiling"
)


try:

    profile = profile_data(
        df
    )

except Exception as e:

    st.error(
        f"Profiling failed: {e}"
    )

    st.stop()


col1, col2 = st.columns(2)


with col1:

    st.write(
        "**Rows:**",
        profile["Total Rows"]
    )

    st.write(
        "**Columns:**",
        profile["Total Columns"]
    )


with col2:

    st.write(
        "**Duplicate Rows:**",
        profile["Duplicate Rows"]
    )

    st.write(
        "**Column Names:**",
        profile["Column Names"]
    )

st.header(
    "Data Preprocessing"
)


if st.button(
    "Run Preprocessing",
    type="primary"
):

    with st.spinner(
        "Preprocessing dataset..."
    ):

        try:

            clean_df, preprocessing_report = (
                preprocess_data(df)
            )

            st.session_state[
                "clean_df"
            ] = clean_df

            st.session_state[
                "preprocessing_report"
            ] = preprocessing_report

            st.session_state.pop(
                "generated_files",
                None
            )

            st.success(
                "Preprocessing completed successfully!"
            )

        except Exception as e:

            st.error(
                f"Preprocessing failed: {e}"
            )

            st.stop()

if "clean_df" not in st.session_state:

    st.info(
        "Run preprocessing to continue."
    )

    st.stop()


clean_df = st.session_state[
    "clean_df"
]


preprocessing_report = st.session_state[
    "preprocessing_report"
]

st.subheader(
    "Preprocessing Report"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Duplicates Removed",
        preprocessing_report.get(
            "removed_duplicate_rows",
            0
        )
    )


with col2:

    st.metric(
        "Empty Rows Removed",
        preprocessing_report.get(
            "removed_empty_rows",
            0
        )
    )


with col3:

    st.metric(
        "Cleaned Rows",
        len(clean_df)
    )

st.subheader(
    "Cleaned Dataset"
)


st.dataframe(
    clean_df.head(20),
    use_container_width=True
)

st.subheader(
    "Download Cleaned Dataset"
)


cleaned_csv = clean_df.to_csv(
    index=False
).encode(
    "utf-8"
)


st.download_button(
    label="Download Cleaned CSV",
    data=cleaned_csv,
    file_name="cleaned_data.csv",
    mime="text/csv"
)

st.header(
    "Automatic Data Visualization"
)


st.write(
    "The visualization engine automatically detects "
    "the types of columns in your dataset and generates "
    "all meaningful chart types that apply."
)

if st.button(
    "Generate All Visualizations",
    type="primary"
):

    output_dir = os.path.join(
        "output",
        "charts"
    )

    if os.path.exists(
        output_dir
    ):

        for file_name in os.listdir(
            output_dir
        ):

            file_path = os.path.join(
                output_dir,
                file_name
            )

            if os.path.isfile(
                file_path
            ):

                try:

                    os.remove(
                        file_path
                    )

                except Exception:
                    pass


    with st.spinner(
        "Generating visualizations..."
    ):

        try:

            generated_files = (
                generate_all_charts(
                    clean_df,
                    output_dir
                )
            )

            st.session_state[
                "generated_files"
            ] = generated_files

            st.success(
                f"Successfully generated "
                f"{len(generated_files)} charts!"
            )

        except Exception as e:

            st.error(
                f"Chart generation failed: {e}"
            )

            st.exception(e)

            st.stop()


if "generated_files" in st.session_state:

    generated_files = st.session_state[
        "generated_files"
    ]


    st.subheader(
        f"Generated Visualizations "
        f"({len(generated_files)})"
    )


    if len(generated_files) == 0:

        st.warning(
            "No visualizations were generated "
            "for this dataset."
        )


    else:

        for index, file_path in enumerate(
            generated_files,
            start=1
        ):

            if os.path.exists(
                file_path
            ):

                st.markdown(
                    f"### Chart {index}: "
                    f"{os.path.basename(file_path)}"
                )

                st.image(
                    file_path,
                    use_container_width=True
                )

                st.divider()

            else:

                st.warning(
                    f"Chart file not found: "
                    f"{file_path}"
                )


st.success(
    "Analysis complete!"
)