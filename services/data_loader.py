
from pathlib import Path

import pandas as pd


def load_dataset(file_path):
    """Load a CSV or Excel file from a path or uploaded file."""

    if hasattr(file_path, "read") and hasattr(file_path, "name"):
        filename = file_path.name.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Please upload a CSV or Excel file.")

    else:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file format.")

    if df.empty or len(df.columns) == 0:
        raise ValueError("The dataset is empty.")

    return df


def preprocess_dataset(df):
    """Clean basic formatting without changing the original."""

    cleaned = df.copy()

    # Remove spaces around column names.
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    if cleaned.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names found after trimming spaces."
        )

    # Trim surrounding spaces in text cells.
    text_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

        cleaned[column] = cleaned[column].replace(
            r"^\s*$", pd.NA, regex=True
        )

    # Remove completely empty rows and columns.
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.dropna(axis=1, how="all")

    return cleaned


def inspect_dataset(df):
    """Summarize the dataset without assuming column names."""

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "missing_values": {
            col: int(count)
            for col, count in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": df.describe(
            include="number"
        ).to_dict(),
    }
