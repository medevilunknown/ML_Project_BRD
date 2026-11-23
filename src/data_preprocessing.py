"""Dataset loading, target construction, feature selection, and splitting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_HINTS = ("calf", "animal", "sender", "id")
DIAGNOSIS_HINTS = ("pneuday", "pneum", "brd", "diagnos", "treat")
LEAKAGE_HINTS = DIAGNOSIS_HINTS + ("status",)


def detect_columns(df: pd.DataFrame) -> dict[str, list[str] | str | None]:
    """Detect likely identifiers and diagnosis fields, reporting all matches."""
    names = {str(c): str(c).lower().replace(" ", "_") for c in df.columns}
    id_matches = [c for c, low in names.items() if any(h in low for h in ID_HINTS)]
    diagnosis_matches = [c for c, low in names.items() if any(h in low for h in DIAGNOSIS_HINTS)]
    calf_id = next((c for c in id_matches if "sender" in names[c]), None)
    if calf_id is None and id_matches:
        calf_id = id_matches[0]
    pneuday = next((c for c in diagnosis_matches if "pneuday" in names[c]), None)
    return {
        "id_candidates": id_matches,
        "diagnosis_candidates": diagnosis_matches,
        "calf_id": calf_id,
        "pneuday": pneuday,
    }


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the first workbook sheet and print a compact inspection summary."""
    path = Path(path)
    df = pd.read_excel(path)
    detected = detect_columns(df)
    print(f"Dataset: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Dtypes:\n", df.dtypes.to_string())
    print("First 5 rows:\n", df.head().to_string())
    print("Missing values:\n", df.isna().sum().sort_values(ascending=False).to_string())
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Detected calf ID candidates: {detected['id_candidates']}")
    print(f"Detected diagnosis candidates: {detected['diagnosis_candidates']}")
    print(f"Selected calf ID: {detected['calf_id']}")
    print(f"Selected diagnosis timing column: {detected['pneuday']}")
    print("Descriptive statistics:\n", df.describe(include="all").transpose().to_string())
    if not detected["calf_id"] or not detected["pneuday"]:
        raise ValueError("Could not safely identify both a calf ID and Pneuday column.")
    return df, detected


def time_to_minutes(value) -> float:
    """Convert an HH:MM:SS duration (datetime.time or string) to minutes."""
    if pd.isna(value):
        return np.nan
    if hasattr(value, "hour"):
        return value.hour * 60 + value.minute + value.second / 60
    try:
        h, m, *s = str(value).strip().split(":")
        return int(h) * 60 + int(m) + (float(s[0]) if s else 0) / 60
    except (ValueError, TypeError):
        return np.nan


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data-type and spelling issues found during exploratory analysis.

    * Lying_time is stored as clock-style durations (e.g. 20:06:46). Left as text it
      would be one-hot encoded into thousands of meaningless columns, so it is
      converted to a numeric number of minutes.
    * PERSON_SCORING contains case/spelling variants of the same scorer
      (Melissa/melissa/Melisa, Megan/megan), which are merged.
    """
    work = df.copy()
    if "Lying_time" in work.columns:
        work["Lying_time"] = work["Lying_time"].map(time_to_minutes).astype(float)
    if "PERSON_SCORING" in work.columns:
        names = work["PERSON_SCORING"].astype("string").str.strip().str.capitalize()
        work["PERSON_SCORING"] = names.replace({"Melisa": "Melissa"}).astype(object).where(names.notna(), np.nan)
    return work


def create_target(df: pd.DataFrame, calf_id: str, pneuday: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep pre-BRD rows and healthy-control rows, then create a binary target."""
    work = clean_columns(df)
    work["_row"] = work.index  # pointer back to the full daily history (used by the 1D-CNN)
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[calf_id]).copy()
    diagnosis_day = pd.to_numeric(work[pneuday], errors="coerce").eq(0)
    diagnosed_calves = set(work.loc[diagnosis_day, calf_id].unique())
    pre_brd = work[pneuday].between(-6, -1, inclusive="both")
    healthy_calves = set(work[calf_id].unique()) - diagnosed_calves
    healthy = work[work[calf_id].isin(healthy_calves)].copy()
    positive = work[pre_brd].copy()
    positive["target"] = 1
    healthy["target"] = 0
    modeled = pd.concat([positive, healthy], ignore_index=True).reset_index(drop=True)
    for column in modeled.select_dtypes(include="object").columns:
        modeled[column] = modeled[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    info = {
        "diagnosed_calves": diagnosed_calves,
        "healthy_calves": healthy_calves,
        "positive_rows": int((modeled["target"] == 1).sum()),
        "negative_rows": int((modeled["target"] == 0).sum()),
        "positive_calves": int(modeled.loc[modeled.target == 1, calf_id].nunique()),
        "healthy_calves_used": int(modeled.loc[modeled.target == 0, calf_id].nunique()),
    }
    print("\nTarget construction: target=1 for -6 <= Pneuday <= -1; target=0 for calves without Pneuday=0.")
    print(f"Number of positive samples: {info['positive_rows']}")
    print(f"Number of negative samples: {info['negative_rows']}")
    print(f"Number of positive calves: {info['positive_calves']}")
    print(f"Number of healthy calves: {info['healthy_calves_used']}")
    return modeled, info


def select_features(df: pd.DataFrame, calf_id: str, pneuday: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Select usable non-leakage columns and separate numeric/categorical features."""
    excluded = []
    id_like = [c for c in df.columns if any(h in str(c).lower().replace(" ", "_") for h in ID_HINTS)]
    for col in df.columns:
        low = str(col).lower().replace(" ", "_")
        if col in (calf_id, "target", "_row") or col in id_like or any(h in low for h in LEAKAGE_HINTS):
            excluded.append(col)
    # Dates and free-text identifiers are excluded; actual behavioural/feeding fields remain.
    for col in df.columns:
        # Herd has a single value ("uk") in this dataset, so it carries no information.
        # Count is a running row number and Observation a record index (bookkeeping, not
        # measurements: Count would let the model recognise individual calves);
        # agefeeder is an exact duplicate of Feederage.
        if (pd.api.types.is_datetime64_any_dtype(df[col]) or col in ("Date", "Enrolldate", "BIRTHDATE", "Count", "Observation", "agefeeder")
                or df[col].nunique(dropna=True) <= 1):
            if col not in excluded:
                excluded.append(col)
    feature_columns = [c for c in df.columns if c not in excluded]
    numeric = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feature_columns if c not in numeric]
    print(f"Excluded leakage/identifier/date columns: {excluded}")
    print(f"Final features used ({len(feature_columns)}): {feature_columns}")
    print(f"Numeric features: {numeric}")
    print(f"Categorical features: {categorical}")
    return feature_columns, numeric, categorical, excluded


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Build preprocessing that is fitted only inside a model pipeline."""
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)], remainder="drop")


def split_by_calf(df: pd.DataFrame, calf_id: str, random_state: int = 42):
    """Perform an 80/20 GroupShuffleSplit and print the leakage check."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df["target"], groups=df[calf_id]))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    train_ids, test_ids = set(train[calf_id]), set(test[calf_id])
    print(f"Train calves: {len(train_ids)}")
    print(f"Test calves: {len(test_ids)}")
    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    print(f"Calf overlap: {train_ids.intersection(test_ids)}")
    print(f"Train class distribution: {train['target'].value_counts().to_dict()}")
    print(f"Test class distribution: {test['target'].value_counts().to_dict()}")
    assert not train_ids.intersection(test_ids)
    return train, test
