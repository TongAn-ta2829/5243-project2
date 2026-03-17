import pandas as pd
import numpy as np


def get_datetime_columns(df):
    return df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()


def get_numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def extract_datetime_features(df, source_col):
    data = df.copy()

    if source_col not in data.columns:
        return data, "Selected datetime column not found."

    if not pd.api.types.is_datetime64_any_dtype(data[source_col]):
        return data, f"Column '{source_col}' is not a datetime column."

    data[f"{source_col}_year"] = data[source_col].dt.year
    data[f"{source_col}_month"] = data[source_col].dt.month
    data[f"{source_col}_day"] = data[source_col].dt.day

    return data, f"Extracted year, month, and day from '{source_col}'."


def create_binned_feature(df, source_col, new_col, bins):
    data = df.copy()

    if source_col not in data.columns:
        return data, "Selected numeric column not found."

    if not pd.api.types.is_numeric_dtype(data[source_col]):
        return data, f"Column '{source_col}' is not numeric."

    if not new_col or not new_col.strip():
        return data, "New column name cannot be empty."

    try:
        bins = int(bins)
    except Exception:
        return data, "Bins must be an integer."

    if bins < 2:
        return data, "Bins must be at least 2."

    try:
        data[new_col] = pd.cut(data[source_col], bins=bins, duplicates="drop").astype(str)
        return data, f"Created binned feature '{new_col}' from '{source_col}'."
    except Exception as e:
        return data, f"Binning failed: {str(e)}"


def create_arithmetic_feature(df, col1, col2, operation, new_col):
    data = df.copy()

    if col1 not in data.columns or col2 not in data.columns:
        return data, "Selected columns not found."

    if not pd.api.types.is_numeric_dtype(data[col1]) or not pd.api.types.is_numeric_dtype(data[col2]):
        return data, "Both selected columns must be numeric."

    if not new_col or not new_col.strip():
        return data, "New column name cannot be empty."

    try:
        if operation == "add":
            data[new_col] = data[col1] + data[col2]
        elif operation == "subtract":
            data[new_col] = data[col1] - data[col2]
        elif operation == "multiply":
            data[new_col] = data[col1] * data[col2]
        elif operation == "divide":
            data[new_col] = data[col1] / data[col2]
        else:
            return data, "Invalid arithmetic operation."

        return data, f"Created '{new_col}' using {col1} {operation} {col2}."
    except Exception as e:
        return data, f"Arithmetic feature creation failed: {str(e)}"


def apply_feature_engineering(df, options):
    data = df.copy()
    logs = []

    if options["extract_datetime"]:
        data, msg = extract_datetime_features(data, options["datetime_col"])
        logs.append(msg)

    if options["create_binned"]:
        data, msg = create_binned_feature(
            data,
            options["binned_source_col"],
            options["binned_new_col"],
            options["binned_bins"],
        )
        logs.append(msg)

    if options["create_arithmetic"]:
        data, msg = create_arithmetic_feature(
            data,
            options["arith_col1"],
            options["arith_col2"],
            options["arith_operation"],
            options["arith_new_col"],
        )
        logs.append(msg)

    log_df = pd.DataFrame({"Step": range(1, len(logs) + 1), "Message": logs})
    return data, log_df

def apply_one_feature_step(df, step):
    data = df.copy()
    step_type = step.get("type")

    if step_type == "datetime":
        data, msg = extract_datetime_features(data, step["datetime_col"])
        new_cols = [
            f"{step['datetime_col']}_year",
            f"{step['datetime_col']}_month",
            f"{step['datetime_col']}_day",
        ]
        return data, msg, new_cols

    if step_type == "binning":
        data, msg = create_binned_feature(
            data,
            step["binned_source_col"],
            step["binned_new_col"],
            step["binned_bins"],
        )
        return data, msg, [step["binned_new_col"]]

    if step_type == "arithmetic":
        data, msg = create_arithmetic_feature(
            data,
            step["arith_col1"],
            step["arith_col2"],
            step["arith_operation"],
            step["arith_new_col"],
        )
        return data, msg, [step["arith_new_col"]]

    return data, "Unknown step type.", []