from __future__ import annotations
import json
import re
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
RANDOM_STATE = 42


# Generic filesystem helpers

def get_project_root() -> Path:
    """
    Resolve the project root folder. This lets the notebooks run whether they are
    opened from the project root or inside a subfolder.
    """
    cwd = Path.cwd()
    candidates = [cwd, cwd.parent, cwd.parent.parent]
    for candidate in candidates:
        if (candidate / "dataset").exists() and (candidate / "outputs").exists():
            return candidate
    return cwd


def ensure_dir(path: Path | str) -> Path:
    """
    Create a directory if it does not already exist and return it as a Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: Path | str) -> None:
    """
    Save a Python dictionary as a JSON file with indentation for readability.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def normalize_column_name(column_name: str) -> str:
    """
    Standardize column names to snake_case for stable downstream processing.
    """
    column_name = str(column_name).strip().lower()
    column_name = re.sub(r"[^a-z0-9]+", "_", column_name)
    column_name = re.sub(r"_+", "_", column_name).strip("_")
    return column_name


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the dataframe with normalized column names.
    """
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def find_dataset_file(dataset_dir: Path | str) -> Path:
    """
    Find the dataset file inside the dataset folder. The notebook expects the user
    to place the Hotel Booking Demand file in the project-level dataset folder.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder was not found: {dataset_dir}")

    supported_patterns = ["*.csv", "*.xlsx", "*.xls"]
    files = []
    for pattern in supported_patterns:
        files.extend(dataset_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            "No supported dataset file was found in the dataset folder. "
            "Place the Hotel Booking Demand dataset inside the root-level dataset folder."
        )

    files = sorted(files, key=lambda p: (0 if "hotel" in p.name.lower() else 1, p.name.lower()))
    return files[0]


def load_dataset(dataset_dir: Path | str) -> Tuple[pd.DataFrame, Path]:
    """
    Load the first supported file from the dataset folder.
    """
    file_path = find_dataset_file(dataset_dir)

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    df = normalize_dataframe_columns(df)
    return df, file_path


# Dataset understanding, anomaly detection, and visualization

def detect_dataset_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect common anomalies and quality issues in the Hotel Booking Demand dataset.
    """
    anomalies: Dict[str, Any] = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_total": int(df.isna().sum().sum()),
        "high_missing_columns": {},
        "potential_impossible_rows": {},
        "negative_numeric_counts": {},
        "high_cardinality_columns": {},
    }

    missing_ratio = (df.isna().mean() * 100).sort_values(ascending=False)
    anomalies["high_missing_columns"] = {
        col: round(val, 2) for col, val in missing_ratio[missing_ratio > 20].to_dict().items()
    }

    numeric_df = df.select_dtypes(include=np.number)
    for col in numeric_df.columns:
        anomalies["negative_numeric_counts"][col] = int((numeric_df[col] < 0).sum())

    for col in df.select_dtypes(include=["object", "category"]).columns:
        unique_count = int(df[col].nunique(dropna=True))
        if unique_count > 15:
            anomalies["high_cardinality_columns"][col] = unique_count

    guest_cols = [col for col in ["adults", "children", "babies"] if col in df.columns]
    if guest_cols:
        guest_totals = df[guest_cols].fillna(0).sum(axis=1)
        anomalies["potential_impossible_rows"]["zero_total_guests"] = int((guest_totals == 0).sum())

    if "adr" in df.columns:
        anomalies["potential_impossible_rows"]["negative_adr"] = int((df["adr"] < 0).sum())

    if "is_canceled" in df.columns:
        target_counts = df["is_canceled"].value_counts(dropna=False).to_dict()
        anomalies["target_distribution"] = {str(k): int(v) for k, v in target_counts.items()}

    return anomalies


def dataset_overview_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build summary tables for types, missing values, numeric stats, and categorical stats.
    """
    dtype_table = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean() * 100).round(2).values,
        "unique_count": df.nunique(dropna=True).values,
    }).sort_values(["missing_percent", "unique_count"], ascending=[False, False])

    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        numeric_summary = numeric_df.describe().T
        numeric_summary["missing_count"] = numeric_df.isna().sum()
        numeric_summary["missing_percent"] = (numeric_df.isna().mean() * 100).round(2)
        numeric_summary["skew"] = numeric_df.skew(numeric_only=True)
    else:
        numeric_summary = pd.DataFrame()

    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    categorical_rows = []
    for col in categorical_cols:
        mode_series = df[col].mode(dropna=True)
        top_value = mode_series.iloc[0] if not mode_series.empty else np.nan
        top_frequency = int(df[col].value_counts(dropna=True).iloc[0]) if df[col].dropna().shape[0] > 0 else 0
        categorical_rows.append({
            "column": col,
            "missing_count": int(df[col].isna().sum()),
            "missing_percent": round(df[col].isna().mean() * 100, 2),
            "unique_count": int(df[col].nunique(dropna=True)),
            "top_value": str(top_value),
            "top_frequency": top_frequency,
        })
    categorical_summary = pd.DataFrame(categorical_rows).sort_values(
        ["missing_percent", "unique_count"], ascending=[False, False]
    ) if categorical_rows else pd.DataFrame()

    return {
        "dtype_summary": dtype_table,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }


def _save_figure(output_path: Path, tight: bool = True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close()


def create_dataset_visualizations(df: pd.DataFrame, output_dir: Path | str, prefix: str = "dataset", full: bool = True) -> None:
    """
    Save a broad EDA bundle with plots for missing values, target balance, distributions,
    correlations, and selected categorical relationships.
    """
    output_dir = ensure_dir(output_dir)

    # Missing values bar chart
    missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    missing_percent[missing_percent > 0].plot(kind="bar")
    plt.title("Missing Values Percentage by Feature")
    plt.ylabel("Missing Percentage")
    plt.xlabel("Feature")
    _save_figure(output_dir / f"{prefix}_missing_values.png")

    # Dtype distribution
    dtype_counts = df.dtypes.astype(str).value_counts()
    plt.figure(figsize=(8, 5))
    plt.pie(dtype_counts.values, labels=dtype_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Distribution of Column Data Types")
    _save_figure(output_dir / f"{prefix}_dtype_distribution.png")

    # Target distribution
    if "is_canceled" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="is_canceled")
        plt.title("Target Class Distribution: is_canceled")
        plt.xlabel("Cancellation Status")
        plt.ylabel("Count")
        _save_figure(output_dir / f"{prefix}_target_distribution.png")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "is_canceled"]

    # Histograms
    if numeric_cols:
        top_numeric = numeric_cols[: min(12, len(numeric_cols))]
        df[top_numeric].hist(figsize=(16, 12), bins=30)
        plt.suptitle("Numerical Feature Distributions", y=1.02)
        _save_figure(output_dir / f"{prefix}_numeric_histograms.png")

        plt.figure(figsize=(14, max(6, len(top_numeric) * 0.5)))
        sns.boxplot(data=df[top_numeric], orient="h")
        plt.title("Boxplots for Selected Numerical Features")
        _save_figure(output_dir / f"{prefix}_numeric_boxplots.png")

        # Save one boxplot per selected numerical feature for easier interpretation.
        for column in top_numeric:
            plt.figure(figsize=(9, 4.5))
            sns.boxplot(x=df[column])
            plt.title(f"Boxplot of {column}")
            plt.xlabel(column)
            safe_column_name = normalize_column_name(column)
            _save_figure(output_dir / f"{prefix}_numeric_boxplot_{safe_column_name}.png")

        corr = df[["is_canceled"] + top_numeric].corr(numeric_only=True) if "is_canceled" in df.columns else df[top_numeric].corr(numeric_only=True)
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        _save_figure(output_dir / f"{prefix}_correlation_heatmap.png")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if categorical_cols:
        cat_cardinality = df[categorical_cols].nunique().sort_values(ascending=False).head(15)
        plt.figure(figsize=(12, 6))
        cat_cardinality.plot(kind="bar")
        plt.title("Top Categorical Features by Cardinality")
        plt.ylabel("Unique Values")
        plt.xlabel("Feature")
        _save_figure(output_dir / f"{prefix}_categorical_cardinality.png")

    # Domain-oriented plots for the hotel dataset
    if full and {"hotel", "is_canceled"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        cancellation_by_hotel = pd.crosstab(df["hotel"], df["is_canceled"], normalize="index")
        cancellation_by_hotel.plot(kind="bar", stacked=True)
        plt.title("Cancellation Share by Hotel Type")
        plt.ylabel("Proportion")
        plt.legend(title="is_canceled")
        _save_figure(output_dir / f"{prefix}_hotel_vs_cancellation.png")

    if full and {"arrival_date_month", "is_canceled"}.issubset(df.columns):
        monthly_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        month_series = df["arrival_date_month"].astype(str)
        order = [m for m in monthly_order if m in month_series.unique().tolist()] or sorted(month_series.unique().tolist())
        plt.figure(figsize=(12, 5))
        sns.countplot(data=df, x="arrival_date_month", hue="is_canceled", order=order)
        plt.title("Bookings and Cancellations by Arrival Month")
        plt.xticks(rotation=45)
        _save_figure(output_dir / f"{prefix}_bookings_by_month.png")

    if full and {"lead_time", "is_canceled"}.issubset(df.columns):
        plt.figure(figsize=(10, 5))
        sns.kdeplot(data=df.sample(min(25000, len(df)), random_state=RANDOM_STATE), x="lead_time", hue="is_canceled", fill=True, common_norm=False)
        plt.title("Lead Time Distribution by Cancellation Status")
        _save_figure(output_dir / f"{prefix}_lead_time_by_target.png")

    if full and {"adr", "is_canceled"}.issubset(df.columns):
        sample_df = df.sample(min(20000, len(df)), random_state=RANDOM_STATE)
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=sample_df, x="is_canceled", y="adr")
        plt.title("ADR Distribution by Cancellation Status")
        _save_figure(output_dir / f"{prefix}_adr_by_target.png")

    if full and {"market_segment", "is_canceled"}.issubset(df.columns):
        segment_cancel = (
            df.groupby("market_segment")["is_canceled"].mean().sort_values(ascending=False).head(12)
        )
        plt.figure(figsize=(12, 5))
        sns.barplot(x=segment_cancel.index, y=segment_cancel.values)
        plt.title("Cancellation Rate by Market Segment")
        plt.ylabel("Cancellation Rate")
        plt.xticks(rotation=45)
        _save_figure(output_dir / f"{prefix}_market_segment_cancel_rate.png")


def create_dataset_description_bundle(
    df: pd.DataFrame,
    output_dir: Path | str,
    prefix: str = "dataset",
    full: bool = True,
) -> Dict[str, Any]:
    """
    Generate summary tables, anomaly reports, and EDA visualizations.
    """
    output_dir = ensure_dir(output_dir)
    overview = dataset_overview_tables(df)
    anomalies = detect_dataset_anomalies(df)

    overview["dtype_summary"].to_csv(output_dir / f"{prefix}_dtype_summary.csv", index=False)
    if not overview["numeric_summary"].empty:
        overview["numeric_summary"].to_csv(output_dir / f"{prefix}_numeric_summary.csv")
    if not overview["categorical_summary"].empty:
        overview["categorical_summary"].to_csv(output_dir / f"{prefix}_categorical_summary.csv", index=False)
    save_json(anomalies, output_dir / f"{prefix}_anomalies_report.json")

    high_level = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "numeric_columns": int(df.select_dtypes(include=np.number).shape[1]),
        "categorical_columns": int(df.select_dtypes(include=["object", "category", "bool"]).shape[1]),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
    }
    save_json(high_level, output_dir / f"{prefix}_overview.json")

    create_dataset_visualizations(df, output_dir=output_dir, prefix=prefix, full=full)
    return {
        "overview": high_level,
        "anomalies": anomalies,
        "tables": overview,
    }