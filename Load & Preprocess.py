from shiny import App, ui, reactive, render
from shiny.render import DataGrid
import pandas as pd
import numpy as np
import json

from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    import pyreadr
    HAS_RDS = True
except Exception:
    HAS_RDS = False


# =========================================================
# DATA LOADING
# =========================================================

def add_missing_values_for_demo(df, frac=0.05, random_state=42):
    """Add a small amount of missing values to numeric columns for demo/testing."""
    out = df.copy()
    rng = np.random.default_rng(random_state)

    num_cols = out.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) == 0:
        return out

    n_rows = len(out)
    n_missing = max(1, int(n_rows * frac))

    for col in num_cols:
        idx = rng.choice(out.index.to_numpy(), size=min(n_missing, n_rows), replace=False)
        out.loc[idx, col] = np.nan

    return out


def create_messy_demo_dataset():
    data = {
        "Patient_ID": [
            "P001", "P002", "P003", "P004", "P005",
            "P006", "P007", "P008", "P008", "P010"
        ],
        "Patient_Age": [
            "12", " 15", "NA", "18", "200",
            "17", "16", "?", "16", "14 "
        ],
        "Gender": [
            "Male", " female", "FEMALE ", "M", "Unknown",
            "male", "Female", "f", "f", ""
        ],
        "Diagnosis_Date": [
            "2024-01-05", "01/10/2024", "2024/01/15", "15-01-2024", "unknown",
            "2024-02-01", " 2024-02-05 ", "2024-02-10", "2024-02-10", "N/A"
        ],
        "Inherited_Father": [
            "Yes", "yes", "Y", "No", "false",
            "1", "0", "No ", "No ", "n"
        ],
        "Maternal_Gene": [
            "No", " no", "N", "Yes", "true",
            "0", "1", "Yes ", "Yes ", "y"
        ],
        "Blood_Cell_mcL": [
            4.8, 5.1, np.nan, 4.9, 50.0,
            4.7, 4.6, 4.8, 4.8, 4.9
        ],
        "Mother_Age": [
            "34", " 29", "31", "unknown", "45",
            "28", "27", "27", "27", "NA"
        ],
        "Father_Age": [
            "40", "38", "N/A", "41", "39",
            "120", "36", "36", "36", " 35 "
        ],
        "Symptom_Score": [
            "2", "3", "4", "missing", "5",
            "2", "3", "4", "4", "?"
        ],
        "Notes": [
            "Stable", " stable ", "Follow-up", "FOLLOW-UP", "None",
            "missing", "Urgent", "urgent ", "urgent ", "?"
        ]
    }

    return pd.DataFrame(data)


def load_builtin_dataset(name):
    if name == "messy_demo":
        return create_messy_demo_dataset()

    if name == "iris":
        df = load_iris(as_frame=True).frame.copy()
        return add_missing_values_for_demo(df, frac=0.05, random_state=42)

    if name == "wine":
        df = load_wine(as_frame=True).frame.copy()
        return add_missing_values_for_demo(df, frac=0.05, random_state=42)

    return None


def read_uploaded_file(file):
    path = file[0]["datapath"]
    name = file[0]["name"].lower()

    if name.endswith(".csv"):
        return pd.read_csv(path)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(path)

    if name.endswith(".json"):
        try:
            return pd.read_json(path)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    return pd.json_normalize(obj)

    if name.endswith(".rds") and HAS_RDS:
        result = pyreadr.read_r(path)
        key = list(result.keys())[0]
        obj = result[key]
        if isinstance(obj, pd.DataFrame):
            return obj
        return pd.DataFrame(obj)

    raise ValueError("Unsupported file format")


# =========================================================
# HELPERS
# =========================================================

def dataset_overview(df):
    return pd.DataFrame(
        {
            "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
            "Value": [
                df.shape[0],
                df.shape[1],
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
            ],
        }
    )


def column_summary(df):
    rows = []
    for col in df.columns:
        rows.append(
            {
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Missing": int(df[col].isna().sum()),
                "Unique Values": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def preview_with_row_numbers(df, n=20):
    preview_df = df.head(n).reset_index(drop=True)
    preview_df.insert(0, "Row", np.arange(1, len(preview_df) + 1))
    return preview_df


def standardize_data_formats(df):
    data = df.copy()

    missing_tokens = {
        "", " ", "na", "n/a", "null", "none", "nan", "missing", "unknown", "?"
    }

    yes_map = {
        "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes"
    }
    no_map = {
        "no": "No", "n": "No", "false": "No", "0": "No"
    }
    gender_map = {
        "m": "Male",
        "male": "Male",
        "f": "Female",
        "female": "Female"
    }

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

            data[col] = data[col].apply(
                lambda x: np.nan
                if isinstance(x, str) and x.strip().lower() in missing_tokens
                else x
            )

            def standardize_text_value(x):
                if not isinstance(x, str):
                    return x

                x_clean = x.strip().lower()

                if x_clean in yes_map:
                    return yes_map[x_clean]
                if x_clean in no_map:
                    return no_map[x_clean]
                if x_clean in gender_map:
                    return gender_map[x_clean]

                return x.strip().title()

            data[col] = data[col].apply(standardize_text_value)

            non_null_original = data[col].notna().sum()

            converted_numeric = pd.to_numeric(data[col], errors="coerce")
            non_null_numeric = converted_numeric.notna().sum()

            if non_null_original > 0 and non_null_numeric / non_null_original >= 0.8:
                data[col] = converted_numeric
                continue

            converted_date = pd.to_datetime(data[col], errors="coerce")
            non_null_date = converted_date.notna().sum()

            if non_null_original > 0 and non_null_date / non_null_original >= 0.8:
                data[col] = converted_date

    return data


def handle_missing_values(df, numeric_method, categorical_method):
    data = df.copy()

    num_cols = data.select_dtypes(include=np.number).columns
    cat_cols = data.select_dtypes(exclude=np.number).columns

    if numeric_method == "mean":
        for col in num_cols:
            data[col] = data[col].fillna(data[col].mean())

    elif numeric_method == "median":
        for col in num_cols:
            data[col] = data[col].fillna(data[col].median())

    elif numeric_method == "zero":
        for col in num_cols:
            data[col] = data[col].fillna(0)

    if categorical_method == "mode":
        for col in cat_cols:
            mode_vals = data[col].mode(dropna=True)
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            data[col] = data[col].fillna(fill_val)

    elif categorical_method == "unknown":
        for col in cat_cols:
            data[col] = data[col].fillna("Unknown")

    return data


def scale_numeric_features(df, method):
    data = df.copy()

    if method == "none":
        return data

    num_cols = data.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) == 0:
        return data

    if method == "standard":
        scaler = StandardScaler()
        data[num_cols] = scaler.fit_transform(data[num_cols])

    elif method == "minmax":
        scaler = MinMaxScaler()
        data[num_cols] = scaler.fit_transform(data[num_cols])

    return data


def encode_categorical_features(df, method):
    data = df.copy()

    if method == "none":
        return data

    cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    if len(cat_cols) == 0:
        return data

    if method == "onehot":
        data = pd.get_dummies(data, columns=cat_cols, drop_first=False)

    elif method == "label":
        for col in cat_cols:
            data[col] = data[col].astype("category").cat.codes

    return data


def handle_outliers(df, method):
    data = df.copy()

    if method == "none":
        return data

    num_cols = data.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) == 0:
        return data

    if method == "cap":
        for col in num_cols:
            series = data[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            data[col] = series.clip(lower=lower, upper=upper)

    elif method == "remove":
        keep_mask = pd.Series(True, index=data.index)
        for col in num_cols:
            series = data[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            keep_mask &= series.isna() | ((series >= lower) & (series <= upper))
        data = data.loc[keep_mask].copy()

    return data


def clean_data(df, options):
    data = df.copy()

    if options["standardize_formats"]:
        data = standardize_data_formats(data)

    if options["remove_duplicates"]:
        data = data.drop_duplicates()

    if options["drop_na_rows"]:
        data = data.dropna()

    data = handle_missing_values(
        data,
        numeric_method=options["numeric_missing"],
        categorical_method=options["categorical_missing"],
    )

    data = handle_outliers(data, options["outlier_method"])
    data = scale_numeric_features(data, options["scale_method"])
    data = encode_categorical_features(data, options["encoding_method"])

    return data


def cleaning_log(before_df, after_df, options):
    return pd.DataFrame(
        {
            "Step": [
                "Rows before",
                "Rows after",
                "Columns before",
                "Columns after",
                "Missing before",
                "Missing after",
                "Duplicates before",
                "Duplicates after",
                "Standardize formats",
                "Remove duplicates",
                "Drop NA rows",
                "Numeric missing strategy",
                "Categorical missing strategy",
                "Scaling method",
                "Encoding method",
                "Outlier handling",
            ],
            "Value": [
                before_df.shape[0],
                after_df.shape[0],
                before_df.shape[1],
                after_df.shape[1],
                int(before_df.isna().sum().sum()),
                int(after_df.isna().sum().sum()),
                int(before_df.duplicated().sum()),
                int(after_df.duplicated().sum()),
                options["standardize_formats"],
                options["remove_duplicates"],
                options["drop_na_rows"],
                options["numeric_missing"],
                options["categorical_missing"],
                options["scale_method"],
                options["encoding_method"],
                options["outlier_method"],
            ],
        }
    )


# =========================================================
# UI
# =========================================================

app_ui = ui.page_fluid(
    ui.h1("Project 2 – Interactive Data Processing Application"),

    ui.navset_tab(
        
        ui.nav_panel(
    "User Guide",

    ui.h2("User Guide"),

    ui.p(
        """
Welcome to the Interactive Data Processing Application. This web app is designed
to help users load, inspect, clean, preprocess, and prepare datasets through an
interactive interface. The application follows a structured workflow so that users
can move from raw data input to analysis-ready data output in a clear and efficient way.
"""
    ),

    ui.p(
        """
The app supports key steps in a typical data processing pipeline, including dataset
loading, data cleaning, preprocessing, feature transformation, and exploratory analysis.
Its interface is organized into easy-to-follow sections so that users can understand
their data, apply changes interactively, and immediately view the updated results.
"""
    ),

    ui.hr(),

    ui.h2("Application Structure"),

    ui.tags.ul(
        ui.tags.li("User Guide: explains the purpose of the application and the main workflow."),
        ui.tags.li("1. Loading Datasets: upload a dataset or select a built-in example dataset."),
        ui.tags.li("2. Data Cleaning & Preprocessing: clean and transform the dataset using interactive controls."),
        ui.tags.li("3. Feature Engineering: prepare variables for downstream analysis and modeling."),
        ui.tags.li("4. Exploratory Data Analysis: inspect distributions, patterns, and relationships in the data."),
    ),

    ui.hr(),

    ui.h2("How to Use the Application"),

    ui.h3("Step 1. Load a Dataset"),

    ui.p(
        """
Go to the “1. Loading Datasets” tab. Users may either upload their own dataset
or select a built-in dataset from the dropdown menu.
"""
    ),

    ui.tags.ul(
        ui.tags.li("Supported upload formats: CSV, Excel (.xlsx, .xls), JSON, and RDS."),
        ui.tags.li("Built-in datasets include example datasets for testing and demonstration."),
        ui.tags.li("If a file is uploaded, the app reads the uploaded file directly."),
        ui.tags.li("If no file is uploaded, users can still explore the application with a built-in dataset."),
    ),

    ui.p(
        """
After a dataset is loaded, the application automatically generates a summary of the data.
This helps users quickly understand the dataset before making any modifications.
"""
    ),

    ui.tags.ul(
        ui.tags.li("Dataset Overview: shows the number of rows, columns, missing values, and duplicate rows."),
        ui.tags.li("Column Summary: shows each variable’s data type, number of missing values, and number of unique values."),
        ui.tags.li("Data Preview: displays the first 20 rows of the dataset with row numbers for readability."),
    ),

    ui.p(
        """
These outputs provide a quick snapshot of both the structure and the quality of the dataset.
"""
    ),

    ui.hr(),

    ui.h3("Step 2. Clean and Preprocess the Data"),

    ui.p(
        """
Go to the “2. Data Cleaning & Preprocessing” tab. This section allows users
to interactively apply common cleaning and preprocessing steps through the sidebar controls.
The results update automatically so users can immediately see the effect of each option.
"""
    ),

    ui.p(
        """
The available preprocessing operations include:
"""
    ),

    ui.tags.ul(
        ui.tags.li("Clean inconsistencies and standardize data formats"),
        ui.tags.li("Remove duplicate rows"),
        ui.tags.li("Drop rows containing missing values"),
        ui.tags.li("Impute missing values"),
        ui.tags.li("Scale numeric features"),
        ui.tags.li("Encode categorical features"),
        ui.tags.li("Handle outliers"),
        ui.tags.li("Download the cleaned dataset"),
    ),

    ui.hr(),

    ui.h3("Detailed Explanation of Preprocessing Options"),

    ui.h4("1. Clean inconsistencies and standardize data formats"),

    ui.p(
        """
This option improves data quality by correcting common formatting issues across columns.
It is especially useful for messy real-world datasets that contain mixed text styles,
placeholder values, or inconsistent data types.
"""
    ),

    ui.tags.ul(
        ui.tags.li("Removes extra spaces from text values."),
        ui.tags.li("Converts tokens such as '', 'NA', 'N/A', 'null', 'missing', 'unknown', and '?' into proper missing values."),
        ui.tags.li("Standardizes Yes/No style values such as yes/y/true/1 and no/n/false/0."),
        ui.tags.li("Standardizes gender values such as M/male and F/female."),
        ui.tags.li("Applies consistent text formatting where appropriate."),
        ui.tags.li("Attempts to convert columns to numeric format when most values are numeric."),
        ui.tags.li("Attempts to convert columns to datetime format when most values represent dates."),
    ),

    ui.h4("2. Remove duplicate rows"),

    ui.p(
        """
If selected, the application removes fully duplicated rows from the dataset.
This reduces redundancy and helps ensure that repeated records do not distort the analysis.
"""
    ),

    ui.h4("3. Drop rows with missing values"),

    ui.p(
        """
If selected, rows that contain missing values are removed from the dataset.
This option is useful when users want to keep only complete observations,
although it may reduce the total number of rows.
"""
    ),

    ui.h4("4. Numeric Missing Value Strategy"),

    ui.p(
        """
Users can choose how missing values in numeric columns should be handled.
"""
    ),

    ui.tags.ul(
        ui.tags.li("Do nothing: keep missing values unchanged."),
        ui.tags.li("Mean imputation: replace missing values with the column mean."),
        ui.tags.li("Median imputation: replace missing values with the column median."),
        ui.tags.li("Fill with 0: replace missing values with zero."),
    ),

    ui.p(
        """
Different strategies are suitable for different situations. For example,
median imputation is often more robust when a variable contains extreme values.
"""
    ),

    ui.h4("5. Categorical Missing Value Strategy"),

    ui.p(
        """
Users can also choose how missing values in categorical columns should be handled.
"""
    ),

    ui.tags.ul(
        ui.tags.li("Do nothing: keep missing values unchanged."),
        ui.tags.li("Mode imputation: replace missing values with the most frequent category."),
        ui.tags.li("Fill with 'Unknown': explicitly label missing values as Unknown."),
    ),

    ui.h4("6. Scaling Method"),

    ui.p(
        """
Numeric features can be scaled to make variables more comparable across different ranges.
This is especially useful before applying many machine learning methods.
"""
    ),

    ui.tags.ul(
        ui.tags.li("None: no scaling is applied."),
        ui.tags.li("StandardScaler: transforms numeric variables to have mean 0 and standard deviation 1."),
        ui.tags.li("MinMaxScaler: rescales numeric variables to the range [0, 1]."),
    ),

    ui.h4("7. Categorical Encoding"),

    ui.p(
        """
Categorical variables can be converted into machine-readable numeric representations.
"""
    ),

    ui.tags.ul(
        ui.tags.li("None: keep categorical variables unchanged."),
        ui.tags.li("One-hot encoding: create binary indicator columns for categories."),
        ui.tags.li("Label encoding: assign an integer code to each category."),
    ),

    ui.p(
        """
One-hot encoding is commonly used when categories do not have a natural order,
while label encoding provides a more compact representation.
"""
    ),

    ui.h4("8. Outlier Handling"),

    ui.p(
        """
Outliers are handled using the IQR (interquartile range) rule.
Users can choose how to treat extreme values in numeric columns.
"""
    ),

    ui.tags.ul(
        ui.tags.li("None: no outlier handling is applied."),
        ui.tags.li("Cap outliers (IQR): clip extreme values to the lower and upper IQR boundaries."),
        ui.tags.li("Remove outlier rows (IQR): remove rows containing outlier values."),
    ),

    ui.p(
        """
This helps reduce the influence of extreme observations that may distort summaries
or affect later modeling steps.
"""
    ),

    ui.h4("9. Download Cleaned Dataset"),

    ui.p(
        """
After preprocessing is complete, users can download the cleaned dataset as a CSV file.
This makes it easy to continue analysis, visualization, or modeling outside the application.
"""
    ),

    ui.hr(),

    ui.h3("Outputs in the Cleaning Module"),

    ui.p(
        """
The preprocessing tab displays multiple outputs so users can compare the data
before and after processing.
"""
    ),

    ui.tags.ul(
        ui.tags.li("Original Dataset Preview: shows the dataset before any preprocessing steps are applied."),
        ui.tags.li("Cleaning Log: summarizes the selected operations and shows how the dataset changed."),
        ui.tags.li("Cleaned Dataset Overview: reports the updated number of rows, columns, missing values, and duplicates."),
        ui.tags.li("Cleaned Dataset Preview: shows the first 20 rows of the processed dataset."),
    ),

    ui.p(
        """
These outputs provide real-time feedback and help users understand the impact
of each preprocessing choice.
"""
    ),

    ui.hr(),

    ui.h2("Feature Engineering"),

    ui.p(
        """
Feature engineering supports the creation or transformation of variables so that
the dataset becomes more informative and better suited for analysis or predictive modeling.
Examples include creating grouped variables, extracting information from dates,
building ratios, and generating derived features from existing columns.
"""
    ),

    ui.p(
        """
This step is important because carefully designed features often improve both
interpretability and model performance.
"""
    ),

    ui.hr(),

    ui.h2("Exploratory Data Analysis"),

    ui.p(
        """
Exploratory Data Analysis (EDA) helps users understand distributions, identify unusual
patterns, and examine relationships among variables. Typical outputs in an EDA workflow
include histograms, boxplots, scatterplots, bar charts, and summary statistics.
"""
    ),

    ui.p(
        """
EDA is useful for validating data quality, identifying trends, and guiding later
modeling decisions.
"""
    ),

    ui.hr(),

    ui.h2("User Experience Design"),

    ui.tags.ul(
        ui.tags.li("The application is organized into tabs to support a clear step-by-step workflow."),
        ui.tags.li("Sidebar controls make preprocessing operations easy to select and modify."),
        ui.tags.li("Tables update automatically so users can immediately see the effect of each change."),
        ui.tags.li("Built-in datasets allow users to test the application even without uploading a file."),
        ui.tags.li("The interface is designed to be clear, accessible, and beginner-friendly."),
    ),

    ui.hr(),

    ui.h2("Recommended Workflow"),

    ui.tags.ol(
        ui.tags.li("Open the User Guide to understand the purpose of the app and the overall workflow."),
        ui.tags.li("Go to Loading Datasets and upload a file or choose a built-in dataset."),
        ui.tags.li("Inspect the dataset overview, column summary, and preview."),
        ui.tags.li("Go to Data Cleaning & Preprocessing and choose the desired preprocessing options."),
        ui.tags.li("Review the original preview, cleaning log, and cleaned preview."),
        ui.tags.li("Download the cleaned dataset for further analysis or modeling."),
    ),

    ui.hr(),

    ui.h2("Summary"),

    ui.p(
        """
This application provides an interactive and practical environment for dataset loading,
cleaning, preprocessing, and preparation. By combining automated summaries,
user-controlled transformations, and immediate visual feedback, the app helps users
move efficiently from raw data to a cleaner and more analysis-ready dataset.
"""
    ),
),



        ui.nav_panel(
            "1. Loading Datasets",

            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select(
                        "builtin",
                        "Built-in Dataset",
                        {
                            "none": "None",
                            "messy_demo": "Messy Demo Dataset",
                            "iris": "Iris",
                            "wine": "Wine",
                        },
                    ),

                    ui.input_file(
                        "upload",
                        "Upload Dataset",
                        accept=[".csv", ".xlsx", ".xls", ".json", ".rds"],
                    ),

                    ui.output_text_verbatim("load_status"),
                ),

                ui.h3("Dataset Overview"),
                ui.output_data_frame("overview"),

                ui.h3("Column Summary"),
                ui.output_data_frame("summary"),

                ui.h3("Data Preview"),
                ui.output_data_frame("preview"),
            ),
        ),

        ui.nav_panel(
            "2. Data Cleaning & Preprocessing",

            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_checkbox(
                        "standardize_formats",
                        "Clean inconsistencies and standardize data formats",
                        True
                    ),
                    ui.input_checkbox("remove_dup", "Remove duplicate rows", True),
                    ui.input_checkbox("drop_na_rows", "Drop rows with missing values", False),

                    ui.input_select(
                        "numeric_missing",
                        "Numeric Missing Value Strategy",
                        {
                            "none": "Do nothing",
                            "mean": "Mean imputation",
                            "median": "Median imputation",
                            "zero": "Fill with 0",
                        },
                    ),

                    ui.input_select(
                        "categorical_missing",
                        "Categorical Missing Value Strategy",
                        {
                            "none": "Do nothing",
                            "mode": "Mode imputation",
                            "unknown": "Fill with 'Unknown'",
                        },
                    ),

                    ui.input_select(
                        "scale_method",
                        "Scaling Method",
                        {
                            "none": "None",
                            "standard": "StandardScaler",
                            "minmax": "MinMaxScaler",
                        },
                    ),

                    ui.input_select(
                        "encoding_method",
                        "Categorical Encoding",
                        {
                            "none": "None",
                            "onehot": "One-hot encoding",
                            "label": "Label encoding",
                        },
                    ),

                    ui.input_select(
                        "outlier_method",
                        "Outlier Handling",
                        {
                            "none": "None",
                            "cap": "Cap outliers (IQR)",
                            "remove": "Remove outlier rows (IQR)",
                        },
                    ),

                    ui.hr(),
                    ui.download_button("download_cleaned", "Download Cleaned Dataset"),
                ),

                ui.h3("Original Dataset Preview"),
                ui.output_data_frame("original_preview"),

                ui.h3("Cleaning Log"),
                ui.output_data_frame("clean_log"),

                ui.h3("Cleaned Dataset Overview"),
                ui.output_data_frame("cleaned_overview"),

                ui.h3("Cleaned Dataset Preview"),
                ui.output_data_frame("cleaned"),
            ),
        ),
    ),
)


# =========================================================
# SERVER
# =========================================================

def server(input, output, session):

    @reactive.calc
    def raw_data():
        if input.upload() is not None:
            try:
                return read_uploaded_file(input.upload())
            except Exception:
                return None

        if input.builtin() != "none":
            return load_builtin_dataset(input.builtin())

        return None

    @output
    @render.text
    def load_status():
        df = raw_data()
        if df is None:
            return "No dataset loaded"
        return f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns"

    @output
    @render.data_frame
    def overview():
        df = raw_data()
        if df is None:
            return DataGrid(pd.DataFrame({"Info": ["No dataset loaded"]}))
        return DataGrid(dataset_overview(df), width="100%")

    @output
    @render.data_frame
    def summary():
        df = raw_data()
        if df is None:
            return DataGrid(pd.DataFrame({"Info": ["No dataset loaded"]}))
        return DataGrid(column_summary(df), width="100%")

    @output
    @render.data_frame
    def preview():
        df = raw_data()
        if df is None:
            return DataGrid(pd.DataFrame())
        return DataGrid(preview_with_row_numbers(df), width="100%")

    @output
    @render.data_frame
    def original_preview():
        df = raw_data()
        if df is None:
            return DataGrid(pd.DataFrame())
        return DataGrid(preview_with_row_numbers(df), width="100%")

    @reactive.calc
    def cleaning_options():
        return {
            "standardize_formats": input.standardize_formats(),
            "remove_duplicates": input.remove_dup(),
            "drop_na_rows": input.drop_na_rows(),
            "numeric_missing": input.numeric_missing(),
            "categorical_missing": input.categorical_missing(),
            "scale_method": input.scale_method(),
            "encoding_method": input.encoding_method(),
            "outlier_method": input.outlier_method(),
        }

    @reactive.calc
    def cleaned_data():
        df = raw_data()
        if df is None:
            return None
        return clean_data(df, cleaning_options())

    @output
    @render.data_frame
    def clean_log():
        raw = raw_data()
        cleaned = cleaned_data()

        if raw is None or cleaned is None:
            return DataGrid(pd.DataFrame({"Info": ["No dataset loaded"]}))

        return DataGrid(cleaning_log(raw, cleaned, cleaning_options()), width="100%")

    @output
    @render.data_frame
    def cleaned_overview():
        df = cleaned_data()
        if df is None:
            return DataGrid(pd.DataFrame({"Info": ["No cleaned dataset"]}))
        return DataGrid(dataset_overview(df), width="100%")

    @output
    @render.data_frame
    def cleaned():
        df = cleaned_data()
        if df is None:
            return DataGrid(pd.DataFrame())
        return DataGrid(preview_with_row_numbers(df), width="100%")

    @output
    @render.download(filename="cleaned_dataset.csv")
    def download_cleaned():
        df = cleaned_data()
        if df is None:
            yield ""
        else:
            yield df.to_csv(index=False)


app = App(app_ui, server)

