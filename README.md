# AUTO_DATA_ANALYSIS

AUTO_DATA_ANALYSIS is a data analysis application which will automatically profile, preprocess, visualize, and analyze CSV datasets with minimal manual effort.

## Objective

It turns raw CSV datasets into meaningful insights.

## Profiling

* Dataset shape and structure
* Column data types
* Missing-value analysis
* Duplicate detection
* Numerical statistics
* Categorical information

## Preprocessing

* Missing-value handling
* Duplicate removal
* Data-type processing
* Date/time conversion
* Basic data cleaning

## Visualization

* Numerical distributions
* Categorical visualizations
* Relationship-based charts
* Grouped mean charts
* Categorical heatmaps

## Implementation of AI

### Use of AI

Here I am using `Qwen/Qwen2.5-7B-Instruct`.

It receives and analyzes dataset metadata using information such as:

* Number of rows
* Column names
* Data types
* Number of unique values
* Missing values
* Minimum, maximum, and mean values for numerical columns
* Sample values for categorical columns

It determines which visualizations are useful for the dataset and then selects appropriate chart types.

Basically, Qwen is acting as the decision-making layer for the visualization section.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Aneebon/auto_data_analysis.git
cd auto_data_analysis
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

For Windows:

```bash
venv\Scripts\activate
```

For Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
```

Replace `your_huggingface_api_token` with your Hugging Face API token.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in the browser.

## Usage

1. Run the Streamlit application.
2. Upload a CSV dataset.
3. The application profiles the dataset.
4. The dataset is preprocessed automatically.
5. The cleaned dataset is generated.
6. Qwen analyzes the dataset metadata.
7. Qwen generates a visualization plan.
8. The visualization module generates the recommended charts.
9. The generated charts and analysis results are displayed in the application.
