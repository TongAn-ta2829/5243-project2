import pandas as pd
import numpy as np
import json
from sklearn.datasets import load_iris, load_wine

try:
    import pyreadr
    HAS_RDS = True
except Exception:
    HAS_RDS = False


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