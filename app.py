from shiny import App, ui, reactive, render
from shiny.render import DataGrid
import pandas as pd

from data_loading import load_builtin_dataset, read_uploaded_file
from eda import register_eda_server

from data_cleaning import (
    dataset_overview,
    column_summary,
    preview_with_row_numbers,
    clean_data,
    cleaning_log,
)

from feature_engineering import (
    get_datetime_columns,
    get_numeric_columns,
    apply_feature_engineering,
    apply_one_feature_step,
)

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
Feature engineering supports the creation of new variables derived from the cleaned dataset.
In this app, you can add feature steps one at a time and the app will keep a running “history”
of the steps you’ve added.
"""
    ),

    ui.p(
        """
Supported feature steps include:
"""
    ),

    ui.tags.ul(
        ui.tags.li("Datetime feature extraction: create year/month/day columns from a datetime column."),
        ui.tags.li("Numeric binning: convert a numeric column into bins (categories)."),
        ui.tags.li("Arithmetic features: add/subtract/multiply/divide two numeric columns into a new feature."),
    ),

    ui.p(
        """
Use “Add feature step” to append a step to the history (the log will show each step and any messages).
If you make a mistake, you can remove a step by its step number and the engineered dataset will update.
"""
    ),

    ui.hr(),

    ui.h2("Exploratory Data Analysis"),

    ui.p(
        """
Exploratory Data Analysis (EDA) helps you understand distributions, identify unusual patterns,
and examine relationships among variables. In this tab you can choose which dataset to explore
(raw, cleaned, or feature engineered), select a plot type, and then select appropriate columns.
"""
    ),

    ui.p(
        """
The tab also provides a summary table for the selected dataset (data type, missingness,
unique values, and basic numeric/categorical summaries).
"""
    ),

    ui.p(
        """
Note: plots require the optional Python package “matplotlib”. If it isn’t installed,
the app will still show the summary table and will display instructions for enabling plots.
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
        ui.tags.li("Optionally go to Feature Engineering to create derived features and review the feature log."),
        ui.tags.li("Optionally go to Exploratory Data Analysis to visualize and summarize the dataset."),
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
        ui.nav_panel(
            "3. Feature Engineering",

            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_action_button("add_feature_step", "Add Current Feature Step"),
                    ui.input_numeric("remove_step_number", "Remove Step Number", value=1, min=1),
                    ui.input_action_button("remove_feature_step", "Remove Selected Step"),

                    ui.hr(),
                    ui.h4("Datetime Features"),
                    ui.input_checkbox("extract_datetime", "Extract year/month/day", False),
                    ui.output_ui("datetime_col_ui"),

                    ui.hr(),

                    ui.h4("Numeric Binning"),
                    ui.input_checkbox("create_binned", "Create binned feature", False),
                    ui.output_ui("binned_source_col_ui"),
                    ui.input_text("binned_new_col", "New binned column name", "binned_feature"),
                    ui.input_numeric("binned_bins", "Number of bins", 3, min=2),

                    ui.hr(),

                    ui.h4("Arithmetic Feature"),
                    ui.input_checkbox("create_arithmetic", "Create arithmetic feature", False),
                    ui.output_ui("arith_col1_ui"),
                    ui.output_ui("arith_col2_ui"),
                    ui.input_select(
                        "arith_operation",
                        "Operation",
                        {
                            "add": "Addition",
                            "subtract": "Subtraction",
                            "multiply": "Multiplication",
                            "divide": "Division",
                        },
                    ),
                    ui.input_text("arith_new_col", "New arithmetic column name", "new_feature"),
                    


                ),

                ui.h3("Feature Engineering Log"),
                ui.output_data_frame("feature_log"),

                ui.h3("Features Created"),
                ui.output_text_verbatim("features_created"),

                ui.h3("Engineered Dataset Preview"),
                ui.output_data_frame("featured_preview"),

                ui.h3("Engineered Dataset Overview"),
                ui.output_data_frame("featured_overview"),

                ui.output_text_verbatim("feature_warning"),
            ),
        ),
        ui.nav_panel(
            "4. Exploratory Data Analysis",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select(
                        "eda_source",
                        "Dataset to explore",
                        {
                            "raw": "Raw (loaded)",
                            "cleaned": "Cleaned",
                            "featured": "Feature engineered",
                        },
                        selected="cleaned",
                    ),
                    ui.input_select(
                        "eda_plot_type",
                        "Plot type",
                        {
                            "hist": "Histogram (numeric)",
                            "box": "Boxplot (numeric)",
                            "scatter": "Scatter (numeric vs numeric)",
                            "bar": "Bar chart (categorical counts)",
                        },
                        selected="hist",
                    ),
                    ui.output_ui("eda_col_ui"),
                    ui.output_ui("eda_x_ui"),
                    ui.output_ui("eda_y_ui"),
                    ui.input_numeric("eda_bins", "Histogram bins", 20, min=2),
                    ui.input_numeric("eda_top_n", "Top N categories (bar)", 20, min=1),
                    ui.input_checkbox("eda_dropna", "Drop missing values for plot", True),
                ),
                ui.h3("EDA Plot"),
                ui.output_text_verbatim("eda_plot_note"),
                ui.output_plot("eda_plot"),
                ui.hr(),
                ui.h3("Summary (selected dataset)"),
                ui.output_data_frame("eda_summary"),
            ),
        )



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
    
    feature_history = reactive.value([])

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
    def feature_options():
        return {
            "extract_datetime": input.extract_datetime(),
            "datetime_col": input.datetime_col(),

            "create_binned": input.create_binned(),
            "binned_source_col": input.binned_source_col(),
            "binned_new_col": input.binned_new_col(),
            "binned_bins": input.binned_bins(),

            "create_arithmetic": input.create_arithmetic(),
            "arith_col1": input.arith_col1(),
            "arith_col2": input.arith_col2(),
            "arith_operation": input.arith_operation(),
            "arith_new_col": input.arith_new_col(),
        }
        
    @reactive.effect
    @reactive.event(input.extract_datetime)
    def _when_datetime_checked():
        if input.extract_datetime():
            ui.update_checkbox("create_binned", value=False)
            ui.update_checkbox("create_arithmetic", value=False)


    @reactive.effect
    @reactive.event(input.create_binned)
    def _when_binning_checked():
        if input.create_binned():
            ui.update_checkbox("extract_datetime", value=False)
            ui.update_checkbox("create_arithmetic", value=False)


    @reactive.effect
    @reactive.event(input.create_arithmetic)
    def _when_arithmetic_checked():
        if input.create_arithmetic():
            ui.update_checkbox("extract_datetime", value=False)
            ui.update_checkbox("create_binned", value=False)


    def build_current_feature_step():
        opts = feature_options()

        if opts["extract_datetime"] and opts["datetime_col"]:
            return {
                "type": "datetime",
                "datetime_col": opts["datetime_col"],
            }

        if opts["create_binned"] and opts["binned_source_col"] and opts["binned_new_col"]:
            return {
                "type": "binning",
                "binned_source_col": opts["binned_source_col"],
                "binned_new_col": opts["binned_new_col"],
                "binned_bins": opts["binned_bins"],
            }

        if (
            opts["create_arithmetic"]
            and opts["arith_col1"]
            and opts["arith_col2"]
            and opts["arith_new_col"]
        ):
            return {
                "type": "arithmetic",
                "arith_col1": opts["arith_col1"],
                "arith_col2": opts["arith_col2"],
                "arith_operation": opts["arith_operation"],
                "arith_new_col": opts["arith_new_col"],
            }

        return None
    
    @reactive.calc
    def feature_warning_message():
        step = build_current_feature_step()
        if step is None:
            return ""

        existing_cols = set(cleaned_data().columns) if cleaned_data() is not None else set()

        history = feature_history.get()
        for s in history:
            if s["type"] == "datetime":
                existing_cols.update([
                    f"{s['datetime_col']}_year",
                    f"{s['datetime_col']}_month",
                    f"{s['datetime_col']}_day",
                ])
            elif s["type"] == "binning":
                existing_cols.add(s["binned_new_col"])
            elif s["type"] == "arithmetic":
                existing_cols.add(s["arith_new_col"])

        new_cols = set()
        if step["type"] == "datetime":
            new_cols = {
                f"{step['datetime_col']}_year",
                f"{step['datetime_col']}_month",
                f"{step['datetime_col']}_day",
            }
        elif step["type"] == "binning":
            new_cols = {step["binned_new_col"]}
        elif step["type"] == "arithmetic":
            new_cols = {step["arith_new_col"]}

        overlap = existing_cols.intersection(new_cols)
        if overlap:
            return (
                "Warning: the following column name(s) already exist and may be overwritten:\n"
                + "\n".join(f"- {col}" for col in overlap)
            )

        return ""

    @output
    @render.text
    def feature_warning():
        return feature_warning_message()


    @reactive.calc
    def cleaned_data():
        df = raw_data()
        opts = cleaning_options()
        if df is None:
            return None
        return clean_data(df, opts)
    @output
    @render.ui
    def datetime_col_ui():
        df = cleaned_data()
        choices = []
        if df is not None:
            choices = get_datetime_columns(df)
        return ui.input_select("datetime_col", "Datetime column", choices=choices)


    @output
    @render.ui
    def binned_source_col_ui():
        df = cleaned_data()
        choices = []
        if df is not None:
            choices = get_numeric_columns(df)
        return ui.input_select("binned_source_col", "Numeric column to bin", choices=choices)


    @output
    @render.ui
    def arith_col1_ui():
        df = cleaned_data()
        choices = []
        if df is not None:
            choices = get_numeric_columns(df)
        return ui.input_select("arith_col1", "First numeric column", choices=choices)


    @output
    @render.ui
    def arith_col2_ui():
        df = cleaned_data()
        choices = []
        if df is not None:
            choices = get_numeric_columns(df)
        return ui.input_select("arith_col2", "Second numeric column", choices=choices)

    @reactive.calc
    def featured_result():
        df = cleaned_data()
        if df is None:
            return None, pd.DataFrame({"Info": ["No cleaned dataset"]})

        history = feature_history.get()

        if not history:
            return df, pd.DataFrame({"Info": ["No feature steps added yet"]})

        data = df.copy()
        rows = []

        for i, step in enumerate(history, start=1):
            data, msg, new_cols = apply_one_feature_step(data, step)
            rows.append(
                {
                    "Step": i,
                    "Type": step["type"],
                    "New Columns": ", ".join(new_cols) if new_cols else "",
                    "Message": msg,
                }
            )

        log_df = pd.DataFrame(rows)
        return data, log_df
    
    @reactive.calc
    def featured_data():
        data, _ = featured_result()
        return data

    register_eda_server(
        input,
        output,
        session,
        raw_data=raw_data,
        cleaned_data=cleaned_data,
        featured_data=featured_data,
    )

    @reactive.calc
    def feature_log_data():
        _, log_df = featured_result()
        return log_df
    
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
        def format_datetime_for_display(df):
            out = df.copy()
            for col in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[col]):
                    out[col] = out[col].dt.strftime("%Y-%m-%d")
            return out
        df = cleaned_data()

        if df is None:
            return DataGrid(pd.DataFrame())
        df_display = format_datetime_for_display(df)
        return DataGrid(preview_with_row_numbers(df_display), width="100%")

    @output
    @render.download(filename="cleaned_dataset.csv")
    def download_cleaned():
        df = cleaned_data()
        if df is None:
            yield ""
        else:
            yield df.to_csv(index=False)
            
    @output
    @render.data_frame
    def feature_log():
        log_df = feature_log_data()
        return DataGrid(log_df, width="100%")

    @output
    @render.data_frame
    def featured_preview():
        df = featured_data()
        if df is None:
            return DataGrid(pd.DataFrame())
        return DataGrid(preview_with_row_numbers(df), width="100%")

    @output
    @render.data_frame
    def featured_overview():
        df = featured_data()
        if df is None:
            return DataGrid(pd.DataFrame({"Info": ["No engineered dataset"]}))
        return DataGrid(dataset_overview(df), width="100%")

    @reactive.effect
    @reactive.event(input.add_feature_step)
    def _add_feature_step():
        step = build_current_feature_step()
        if step is None:
            return

        history = feature_history.get().copy()
        history.append(step)
        feature_history.set(history)


    @reactive.effect
    @reactive.event(input.remove_feature_step)
    def _remove_feature_step():
        history = feature_history.get().copy()
        if not history:
            return

        step_num = input.remove_step_number()
        if step_num is None:
            return

        idx = int(step_num) - 1
        if 0 <= idx < len(history):
            history.pop(idx)
            feature_history.set(history)
    
    @output
    @render.text
    def features_created():
        log_df = feature_log_data()

        if log_df.empty or "New Columns" not in log_df.columns:
            return "No features created yet."

        cols = []
        for item in log_df["New Columns"].tolist():
            if isinstance(item, str) and item.strip():
                cols.extend([x.strip() for x in item.split(",") if x.strip()])

        if not cols:
            return "No features created yet."

        lines = ["Features created:"]
        for c in cols:
            lines.append(f"- {c}")

        return "\n".join(lines)
    
app = App(app_ui, server)

