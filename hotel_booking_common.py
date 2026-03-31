from __future__ import annotations
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
RANDOM_STATE = 42


# ------------------------------------------------------------
# Generic filesystem helpers
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Custom preprocessing transformers
# ------------------------------------------------------------

class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Clip numeric values using the IQR rule to reduce the influence of extreme outliers.
    This transformer is especially useful for variables such as ADR and lead time.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X_array = self._to_numpy(X)
        q1 = np.nanpercentile(X_array, 25, axis=0)
        q3 = np.nanpercentile(X_array, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_array = self._to_numpy(X)
        clipped = np.clip(X_array, self.lower_bounds_, self.upper_bounds_)
        return clipped

    @staticmethod
    def _to_numpy(X):
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        if isinstance(X, pd.Series):
            return X.to_frame().to_numpy(dtype=float)
        return np.asarray(X, dtype=float)


def make_one_hot_encoder():
    """
    Create a version-compatible OneHotEncoder.
    """
    try:
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


# ------------------------------------------------------------
# Dataset understanding, anomaly detection, and visualization
# ------------------------------------------------------------

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


def _ensure_figure_drawn(fig) -> None:
    """Bar positions / containers are reliable only after a canvas draw (needed in notebooks)."""
    try:
        fig.canvas.draw()
    except Exception:
        pass


def _bar_rectangle_patches(ax) -> list:
    """Rectangles that belong to bar plots (excludes axes background patch when listed)."""
    rects: list = []
    for container in ax.containers:
        p = getattr(container, "patches", None)
        if p is not None:
            rects.extend(p)
        else:
            try:
                rects.extend(container)
            except TypeError:
                pass
    if rects:
        return [p for p in rects if isinstance(p, mpatches.Rectangle)]
    out = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)]
    if ax.patch is not None and ax.patch in out:
        out.remove(ax.patch)
    return out


def _annotate_vertical_bar_tops(ax, fmt: str, fontsize: int = 8, y_pad_frac: float = 0.16) -> None:
    """Place numeric labels above each vertical bar; clip_on=False so labels survive tight_layout/savefig."""
    _ensure_figure_drawn(ax.figure)
    for patch in _bar_rectangle_patches(ax):
        h = patch.get_height()
        w = patch.get_width()
        if h <= 0 or not np.isfinite(h) or w <= 0:
            continue
        x = patch.get_x() + w / 2
        ax.annotate(
            fmt % h,
            (x, h),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
        )
    y0, y1 = ax.get_ylim()
    if y1 >= y0:
        ax.set_ylim(y0, y1 + (y1 - y0) * y_pad_frac)
    else:
        ax.set_ylim(y1 - (y0 - y1) * y_pad_frac, y0)


def _annotate_stacked_bar_segment_centers(ax, fmt: str, fontsize: int = 8, min_height: float = 0.03) -> None:
    """Label each stacked segment with its height at the segment midpoint."""
    _ensure_figure_drawn(ax.figure)
    for patch in _bar_rectangle_patches(ax):
        h = patch.get_height()
        w = patch.get_width()
        if h < min_height or not np.isfinite(h) or w <= 0:
            continue
        x = patch.get_x() + w / 2
        y = patch.get_y() + h / 2
        ax.annotate(
            fmt % h,
            (x, y),
            ha="center",
            va="center",
            fontsize=fontsize,
            clip_on=False,
        )


def _pie_autopct_counts_and_pct(values):
    """Pie autopct: absolute count and percentage per wedge."""
    total = float(np.sum(values))

    def autopct(pct: float) -> str:
        n = int(round(pct * total / 100.0)) if total else 0
        return f"{n}\n({pct:.1f}%)"

    return autopct


def create_dataset_visualizations(df: pd.DataFrame, output_dir: Path | str, prefix: str = "dataset", full: bool = True) -> None:
    """
    Save a broad EDA bundle with plots for missing values, target balance, distributions,
    correlations, and selected categorical relationships.
    """
    output_dir = ensure_dir(output_dir)

    # Missing values bar chart
    missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    missing_percent[missing_percent > 0].plot(kind="bar", ax=ax)
    plt.title("Missing Values Percentage by Feature")
    plt.ylabel("Missing Percentage")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    _annotate_vertical_bar_tops(ax, fmt="%.2f", fontsize=7)
    _save_figure(output_dir / f"{prefix}_missing_values.png")

    # Dtype distribution
    dtype_counts = df.dtypes.astype(str).value_counts()
    plt.figure(figsize=(8, 5))
    plt.pie(
        dtype_counts.values,
        labels=dtype_counts.index,
        autopct=_pie_autopct_counts_and_pct(dtype_counts.values),
        startangle=90,
    )
    plt.title("Distribution of Column Data Types")
    _save_figure(output_dir / f"{prefix}_dtype_distribution.png")

    # Target distribution
    if "is_canceled" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="is_canceled", ax=ax)
        plt.title("Target Class Distribution: is_canceled")
        plt.xlabel("Cancellation Status")
        plt.ylabel("Count")
        _annotate_vertical_bar_tops(ax, fmt="%.0f", fontsize=9)
        _save_figure(output_dir / f"{prefix}_target_distribution.png")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "is_canceled"]

    # Histograms
    if numeric_cols:
        top_numeric = numeric_cols[: min(12, len(numeric_cols))]
        hist_axes = df[top_numeric].hist(figsize=(16, 12), bins=30)
        plt.suptitle("Numerical Feature Distributions", y=1.02)
        for ax, col in zip(np.atleast_1d(hist_axes).ravel(), top_numeric):
            s = df[col].dropna()
            if len(s) == 0:
                continue
            ax.text(
                0.98,
                0.98,
                f"n={len(s):,}\nmedian={s.median():.4g}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
            )
        _save_figure(output_dir / f"{prefix}_numeric_histograms.png")

        plt.figure(figsize=(14, max(6, len(top_numeric) * 0.5)))
        sns.boxplot(data=df[top_numeric], orient="h")
        plt.title("Boxplots for Selected Numerical Features")
        plt.figtext(0.99, 0.01, f"n={len(df):,} rows", ha="right", fontsize=9)
        _save_figure(output_dir / f"{prefix}_numeric_boxplots.png")

        # Save one boxplot per selected numerical feature for easier interpretation.
        for column in top_numeric:
            plt.figure(figsize=(9, 4.5))
            s = df[column].dropna()
            med = s.median()
            sns.boxplot(x=df[column])
            plt.title(f"Boxplot of {column} (n={len(s):,}, median={med:.4g})")
            plt.xlabel(column)
            safe_column_name = normalize_column_name(column)
            _save_figure(output_dir / f"{prefix}_numeric_boxplot_{safe_column_name}.png")

        corr = df[["is_canceled"] + top_numeric].corr(numeric_only=True) if "is_canceled" in df.columns else df[top_numeric].corr(numeric_only=True)
        plt.figure(figsize=(12, 9))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".3f",
            annot_kws={"size": 7},
        )
        plt.title("Correlation Heatmap")
        _save_figure(output_dir / f"{prefix}_correlation_heatmap.png")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if categorical_cols:
        cat_cardinality = df[categorical_cols].nunique().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(12, 6))
        cat_cardinality.plot(kind="bar", ax=ax)
        plt.title("Top Categorical Features by Cardinality")
        plt.ylabel("Unique Values")
        plt.xlabel("Feature")
        plt.xticks(rotation=45, ha="right")
        _annotate_vertical_bar_tops(ax, fmt="%.0f", fontsize=8)
        _save_figure(output_dir / f"{prefix}_categorical_cardinality.png")

    # Domain-oriented plots for the hotel dataset
    if full and {"hotel", "is_canceled"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        cancellation_by_hotel = pd.crosstab(df["hotel"], df["is_canceled"], normalize="index")
        cancellation_by_hotel.plot(kind="bar", stacked=True, ax=ax)
        plt.title("Cancellation Share by Hotel Type")
        plt.ylabel("Proportion")
        plt.legend(title="is_canceled")
        plt.xticks(rotation=0)
        _annotate_stacked_bar_segment_centers(ax, fmt="%.2f", fontsize=8)
        _save_figure(output_dir / f"{prefix}_hotel_vs_cancellation.png")

    if full and {"arrival_date_month", "is_canceled"}.issubset(df.columns):
        monthly_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        month_series = df["arrival_date_month"].astype(str)
        order = [m for m in monthly_order if m in month_series.unique().tolist()] or sorted(month_series.unique().tolist())
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.countplot(data=df, x="arrival_date_month", hue="is_canceled", order=order, ax=ax)
        plt.title("Bookings and Cancellations by Arrival Month")
        plt.xticks(rotation=45)
        _annotate_vertical_bar_tops(ax, fmt="%.0f", fontsize=6, y_pad_frac=0.22)
        _save_figure(output_dir / f"{prefix}_bookings_by_month.png")

    if full and {"lead_time", "is_canceled"}.issubset(df.columns):
        lead_sample = df.sample(min(25000, len(df)), random_state=RANDOM_STATE)
        counts = lead_sample["is_canceled"].value_counts().sort_index()
        count_str = ", ".join(f"{int(k)}: {int(v):,}" for k, v in counts.items())
        plt.figure(figsize=(10, 5))
        sns.kdeplot(
            data=lead_sample,
            x="lead_time",
            hue="is_canceled",
            fill=True,
            common_norm=False,
        )
        plt.title(f"Lead Time Distribution by Cancellation Status (sample n by class: {count_str})")
        _save_figure(output_dir / f"{prefix}_lead_time_by_target.png")

    if full and {"adr", "is_canceled"}.issubset(df.columns):
        sample_df = df.sample(min(20000, len(df)), random_state=RANDOM_STATE)
        medians = sample_df.groupby("is_canceled", observed=True)["adr"].median()
        med_str = ", ".join(f"{k}: {v:.2f}" for k, v in medians.items())
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=sample_df, x="is_canceled", y="adr")
        plt.title(f"ADR by Cancellation Status (n={len(sample_df):,}; medians {med_str})")
        _save_figure(output_dir / f"{prefix}_adr_by_target.png")

    if full and {"market_segment", "is_canceled"}.issubset(df.columns):
        segment_cancel = (
            df.groupby("market_segment")["is_canceled"].mean().sort_values(ascending=False).head(12)
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(x=segment_cancel.index, y=segment_cancel.values, ax=ax)
        plt.title("Cancellation Rate by Market Segment")
        plt.ylabel("Cancellation Rate")
        plt.xticks(rotation=45)
        _annotate_vertical_bar_tops(ax, fmt="%.3f", fontsize=8)
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


# ------------------------------------------------------------
# Hotel dataset specific preprocessing and feature engineering
# ------------------------------------------------------------

def coerce_target_binary(y: pd.Series) -> pd.Series:
    """
    Convert target values to 0/1 if they are stored as strings or booleans.
    """
    if set(pd.Series(y).dropna().unique()).issubset({0, 1}):
        return y.astype(int)

    mapping = {
        "0": 0,
        "1": 1,
        "no": 0,
        "yes": 1,
        "not canceled": 0,
        "not cancelled": 0,
        "canceled": 1,
        "cancelled": 1,
        "false": 0,
        "true": 1,
    }
    return y.astype(str).str.strip().str.lower().map(mapping).astype(int)


def prepare_hotel_booking_dataframe(df: pd.DataFrame, target_col: str = "is_canceled") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform assignment-friendly preprocessing analysis and domain-aware feature engineering.
    The actual imputation and encoding remain inside the ML pipeline to avoid leakage.
    """
    df = normalize_dataframe_columns(df)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    notes: Dict[str, Any] = {
        "initial_shape": tuple(df.shape),
        "duplicate_rows_removed": 0,
        "dropped_columns": [],
        "engineered_features": [],
        "removed_rows": {},
        "high_missing_columns": {},
        "leakage_columns_removed": [],
    }

    # Remove exact duplicates
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    notes["duplicate_rows_removed"] = duplicate_rows

    # Remove known leakage columns
    leakage_cols = [col for col in ["reservation_status", "reservation_status_date"] if col in df.columns]
    if leakage_cols:
        df = df.drop(columns=leakage_cols)
        notes["leakage_columns_removed"] = leakage_cols
        notes["dropped_columns"].extend(leakage_cols)

    # Drop extremely sparse columns if necessary (company is usually very sparse)
    missing_ratio = df.isna().mean().sort_values(ascending=False)
    notes["high_missing_columns"] = {
        col: round(val * 100, 2) for col, val in missing_ratio[missing_ratio > 0.2].to_dict().items()
    }
    for col in ["company"]:
        if col in df.columns and df[col].isna().mean() > 0.80:
            df = df.drop(columns=[col])
            notes["dropped_columns"].append(col)

    # Fix obvious column types
    if target_col in df.columns:
        df[target_col] = coerce_target_binary(df[target_col])

    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    if "arrival_date_month" in df.columns:
        month_as_str = df["arrival_date_month"].astype(str).str.strip().str.lower()
        month_num = month_as_str.map(month_map)
        if month_num.notna().sum() > 0:
            df["arrival_month_num"] = month_num
            notes["engineered_features"].append("arrival_month_num")

            def month_to_season(m):
                if pd.isna(m):
                    return np.nan
                m = int(m)
                if m in [12, 1, 2]:
                    return "Winter"
                if m in [3, 4, 5]:
                    return "Spring"
                if m in [6, 7, 8]:
                    return "Summer"
                return "Autumn"

            df["arrival_season"] = df["arrival_month_num"].apply(month_to_season)
            notes["engineered_features"].append("arrival_season")

    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(df.columns):
        df["total_nights"] = (
            df["stays_in_weekend_nights"].fillna(0) + df["stays_in_week_nights"].fillna(0)
        )
        notes["engineered_features"].append("total_nights")

    guest_cols = [col for col in ["adults", "children", "babies"] if col in df.columns]
    if guest_cols:
        df["total_guests"] = df[guest_cols].fillna(0).sum(axis=1)
        notes["engineered_features"].append("total_guests")
        zero_guest_count = int((df["total_guests"] == 0).sum())
        if zero_guest_count > 0:
            df = df[df["total_guests"] > 0].copy()
            notes["removed_rows"]["zero_total_guests"] = zero_guest_count

    if {"adults", "children", "babies"}.issubset(df.columns):
        df["is_family"] = (
            ((df["children"].fillna(0) + df["babies"].fillna(0)) > 0).astype(int)
        )
        notes["engineered_features"].append("is_family")

    if {"reserved_room_type", "assigned_room_type"}.issubset(df.columns):
        df["room_type_changed"] = (
            df["reserved_room_type"].astype(str) != df["assigned_room_type"].astype(str)
        ).astype(int)
        notes["engineered_features"].append("room_type_changed")

    if {"previous_cancellations", "previous_bookings_not_canceled"}.issubset(df.columns):
        df["previous_total_bookings"] = (
            df["previous_cancellations"].fillna(0) + df["previous_bookings_not_canceled"].fillna(0)
        )
        notes["engineered_features"].append("previous_total_bookings")

    if "lead_time" in df.columns and "total_nights" in df.columns:
        df["lead_time_per_night"] = df["lead_time"] / df["total_nights"].replace(0, np.nan)
        notes["engineered_features"].append("lead_time_per_night")

    # Replace impossible negatives in known non-negative business variables with NaN
    for col in [
        "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies", "booking_changes", "days_in_waiting_list",
        "adr", "required_car_parking_spaces", "total_of_special_requests",
        "total_nights", "total_guests", "previous_total_bookings", "lead_time_per_night"
    ]:
        if col in df.columns:
            negative_count = int((df[col] < 0).sum()) if pd.api.types.is_numeric_dtype(df[col]) else 0
            if negative_count > 0:
                df.loc[df[col] < 0, col] = np.nan
                notes["removed_rows"][f"negative_values_set_to_nan_{col}"] = negative_count

    # Treat identifier-like columns as categorical if they survive
    for col in ["agent"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64").astype(str).replace("<NA>", np.nan)

    # A clean final index helps with saved predictions and comparisons
    df = df.reset_index(drop=True)
    notes["final_shape"] = tuple(df.shape)
    return df, notes


def get_feature_groups(df: pd.DataFrame, target_col: str = "is_canceled") -> Tuple[List[str], List[str], List[str]]:
    """
    Split features into feature list, numerical feature list, and categorical feature list.
    """
    feature_cols = [col for col in df.columns if col != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols


def build_preprocessor(model_name: str, numeric_cols: List[str], categorical_cols: List[str]):
    """
    Build the algorithm-specific preprocessing transformer.
    """
    model_name = model_name.lower()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper()),
    ]

    if model_name == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", make_one_hot_encoder()),
        ])
    elif model_name == "gradient_boosting":
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
    else:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", make_one_hot_encoder()),
        ])

    numeric_transformer = Pipeline(steps=numeric_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def build_train_test_split(
    df: pd.DataFrame,
    target_col: str = "is_canceled",
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    """
    Create a reproducible stratified split for all notebooks.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ------------------------------------------------------------
# Evaluation and model interpretation helpers
# ------------------------------------------------------------

def specificity_score(y_true, y_pred) -> float:
    """
    Compute specificity from the confusion matrix.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    denominator = (tn + fp)
    return float(tn / denominator) if denominator else 0.0


def get_probabilities(model, X) -> np.ndarray:
    """
    Get class probabilities or calibrated score-like probabilities.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        probabilities = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        probabilities = model.predict(X).astype(float)
    return probabilities


def evaluate_classifier(
    model,
    X_test,
    y_test,
    output_dir: Path | str,
    model_name: str,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluate a fitted classifier, save metrics, predictions, and visualizations.
    """
    output_dir = ensure_dir(output_dir)

    y_prob = get_probabilities(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity": float(specificity_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision_pr_auc": float(average_precision_score(y_test, y_prob)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "log_loss": float(log_loss(y_test, y_prob, labels=[0, 1])),
        "threshold": threshold,
        "test_rows": int(len(y_test)),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    save_json(metrics, output_dir / "metrics_summary.json")

    predictions_df = pd.DataFrame({
        "y_true": pd.Series(y_test).reset_index(drop=True),
        "y_pred": y_pred,
        "y_probability": y_prob,
    })
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    report_text = classification_report(y_test, y_pred, digits=4)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    _save_figure(output_dir / "confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    _save_figure(output_dir / "roc_curve.png")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {metrics['average_precision_pr_auc']:.4f}")
    plt.title(f"{model_name} - Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    _save_figure(output_dir / "precision_recall_curve.png")

    # Probability distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(predictions_df, x="y_probability", hue="y_true", bins=30, element="step", stat="density", common_norm=False)
    plt.title(f"{model_name} - Predicted Probability Distribution")
    _save_figure(output_dir / "predicted_probability_distribution.png")

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.title(f"{model_name} - Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    _save_figure(output_dir / "calibration_curve.png")

    return metrics


def save_cv_results(search_object, output_dir: Path | str) -> pd.DataFrame:
    """
    Save cross-validation results and a visualization of the top runs.
    """
    output_dir = ensure_dir(output_dir)
    cv_df = pd.DataFrame(search_object.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
    cv_df.to_csv(output_dir / "cv_results.csv", index=False)

    top_df = cv_df.head(15).copy()
    top_df["candidate"] = np.arange(1, len(top_df) + 1)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_df, x="candidate", y="mean_test_score")
    plt.title("Top Hyperparameter Candidates by Mean CV Score")
    plt.xlabel("Candidate Rank")
    plt.ylabel("Mean CV Score")
    _save_figure(output_dir / "cv_top_candidates.png")

    return cv_df


def save_feature_insights(model, output_dir: Path | str, model_name: str, top_n: int = 20) -> pd.DataFrame:
    """
    Save coefficients or feature importances for interpretability.
    """
    output_dir = ensure_dir(output_dir)

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return pd.DataFrame()

    importance_df = pd.DataFrame()
    if hasattr(classifier, "coef_"):
        importance_values = classifier.coef_.ravel()
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_values,
            "abs_importance": np.abs(importance_values),
        }).sort_values("abs_importance", ascending=False)
        plot_title = f"{model_name} - Top Coefficients"
    elif hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_values,
            "abs_importance": np.abs(importance_values),
        }).sort_values("abs_importance", ascending=False)
        plot_title = f"{model_name} - Top Feature Importances"
    else:
        return pd.DataFrame()

    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    top_plot = importance_df.head(top_n).sort_values("abs_importance", ascending=True)

    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    sns.barplot(data=top_plot, x="abs_importance", y="feature")
    plt.title(plot_title)
    plt.xlabel("Absolute Importance")
    plt.ylabel("Feature")
    _save_figure(output_dir / "feature_importance.png")
    return importance_df


def save_model_bundle(
    model,
    output_dir: Path | str,
    model_filename: str = "best_model.joblib",
    best_params: Optional[Dict[str, Any]] = None,
    preprocessing_notes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save the trained pipeline and related metadata.
    """
    output_dir = ensure_dir(output_dir)
    joblib.dump(model, output_dir / model_filename)
    if best_params is not None:
        save_json(best_params, output_dir / "best_params.json")
    if preprocessing_notes is not None:
        save_json(preprocessing_notes, output_dir / "preprocessing_notes.json")


def summarize_preprocessing_notes(notes: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert preprocessing notes into a tidy dataframe for display and export.
    """
    rows = []
    for key, value in notes.items():
        rows.append({"item": key, "value": json.dumps(value) if isinstance(value, (dict, list, tuple)) else str(value)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Comparison notebook helpers
# ------------------------------------------------------------

def compare_models_on_test_set(
    model_paths: Dict[str, Path],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path | str,
) -> pd.DataFrame:
    """
    Load multiple saved models, generate predictions on the same test set, and
    save a common comparison bundle.
    """
    output_dir = ensure_dir(output_dir)
    rows = []

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")
    for model_name, model_path in model_paths.items():
        model = joblib.load(model_path)
        y_prob = get_probabilities(model, X_test)
        y_pred = (y_prob >= 0.50).astype(int)

        row = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "specificity": float(specificity_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "f2": float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "average_precision_pr_auc": float(average_precision_score(y_test, y_prob)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
            "brier_score": float(brier_score_loss(y_test, y_prob)),
            "log_loss": float(log_loss(y_test, y_prob, labels=[0, 1])),
        }
        rows.append(row)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={row['roc_auc']:.3f})")

    plt.title("ROC Curve Comparison Across Saved Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    _save_figure(Path(output_dir) / "comparison_roc_curve.png")

    comparison_df = pd.DataFrame(rows).sort_values(["roc_auc", "f1", "mcc"], ascending=False).reset_index(drop=True)
    comparison_df.to_csv(Path(output_dir) / "model_comparison_summary.csv", index=False)

    # Metric bar charts
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc", "balanced_accuracy"]
    for metric_name in metrics_to_plot:
        plt.figure(figsize=(9, 5))
        sns.barplot(data=comparison_df, x="model_name", y=metric_name)
        plt.title(f"Model Comparison - {metric_name.replace('_', ' ').title()}")
        plt.xticks(rotation=15)
        _save_figure(Path(output_dir) / f"comparison_{metric_name}.png")

    # Radar chart
    radar_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for _, row in comparison_df.iterrows():
        values = [row[m] for m in radar_metrics]
        values += values[:1]
        ax.plot(angles, values, label=row["model_name"])
        ax.fill(angles, values, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n").title() for m in radar_metrics])
    ax.set_title("Radar Chart Comparison of Key Metrics")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save_figure(Path(output_dir) / "comparison_radar_chart.png", tight=False)

    best_model = comparison_df.iloc[0].to_dict()
    save_json(best_model, Path(output_dir) / "best_model_summary.json")
    return comparison_df
