import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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

    for col in data.columns:
        if pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_string_dtype(data[col]):
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

                if x_clean in ["m", "male"]:
                    return "Male"
                if x_clean in ["f", "female"]:
                    return "Female"

                if x_clean in ["yes", "y", "true", "1"]:
                    return "Yes"
                if x_clean in ["no", "n", "false", "0"]:
                    return "No"

                return x.strip().title()

            data[col] = data[col].apply(standardize_text_value)

            converted_numeric = pd.to_numeric(data[col], errors="coerce")
            non_null_original = data[col].notna().sum()
            non_null_numeric = converted_numeric.notna().sum()

            if non_null_original > 0 and non_null_numeric >= 3:
                data[col] = converted_numeric
                continue

            def parse_mixed_date(x):
                if pd.isna(x):
                    return pd.NaT

                s = str(x).strip()

                if s == "":
                    return pd.NaT

                if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", s):
                    return pd.to_datetime(s, errors="coerce", yearfirst=True)

                if re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", s):
                    return pd.to_datetime(s, errors="coerce", dayfirst=True)

                if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", s):
                    return pd.to_datetime(s, errors="coerce", format="%m/%d/%Y")

                return pd.to_datetime(s, errors="coerce")

            converted_date = data[col].apply(parse_mixed_date)
            non_null_date = converted_date.notna().sum()

            if non_null_original > 0 and non_null_date >= 3:
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