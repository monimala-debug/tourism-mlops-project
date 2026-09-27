"""
prep.py - Clean the registered dataset, select pre-contact features, and create a
stratified train/test split.

What this script deliberately does NOT do: imputation, scaling or one-hot encoding.
Those steps learn from data, so they live inside the sklearn Pipeline in train.py and
are fitted only on training folds during cross-validation (no test-set leakage).

Run from the repository root:
    python tourism_project/model_building/prep.py
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = Path("tourism_project/data/tourism.csv")
DEFAULT_SPLITS_DIR = Path("tourism_project/data/splits")

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "ProdTaken"

# Columns that are never model inputs.
NON_FEATURE_COLUMNS = ["Unnamed: 0", "CustomerID", TARGET]

# Known only AFTER a salesperson has contacted and pitched the customer -> leakage for a
# model that must score customers BEFORE contact.
POST_CONTACT_COLUMNS = ["DurationOfPitch", "PitchSatisfactionScore", "NumberOfFollowups"]

# Describe the contact/pitch decision itself, which is what the model is meant to inform:
#  - TypeofContact: how the lead was contacted (company invitation vs. self enquiry).
#  - ProductPitched: the package the salesperson chose to pitch. In this file it is a
#    one-to-one relabelling of Designation, so dropping it loses no customer information.
CONTACT_DECISION_COLUMNS = ["TypeofContact", "ProductPitched"]

# Final pre-contact feature set (order is preserved in the split files and the app).
NUMERIC_FEATURES = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = ["Occupation", "Gender", "MaritalStatus", "Designation"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Reproducible label clean-up. Only unambiguous spelling variants are merged.
LABEL_FIXES = {
    "Gender": {"Fe Male": "Female"},            # typo of Female
    "Occupation": {"Free Lancer": "Freelancer"},  # spelling variant
    # MaritalStatus "Unmarried" is intentionally NOT merged into "Single": it may mean
    # "in a relationship but not married", and its purchase rate differs from "Single".
}


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace in text columns and apply LABEL_FIXES. Returns a new frame."""
    df = df.copy()
    text_cols = [c for c in df.columns
                 if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    for col in text_cols:
        df[col] = df[col].str.strip()
    for col, mapping in LABEL_FIXES.items():
        if col in df.columns:
            before = int(df[col].isin(mapping.keys()).sum())
            df[col] = df[col].replace(mapping)
            print(f"Cleaned {col}: {mapping} ({before} rows relabelled)")
    return df


def describe_split(name: str, y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    print(f"{name:<6} rows={len(y):5d}  class 0={counts.get(0, 0):5d}  "
          f"class 1={counts.get(1, 0):4d}  positive rate={y.mean():.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/test splits")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    print(f"Loaded {args.data_path}: {df.shape[0]} rows x {df.shape[1]} columns")

    df = clean_categories(df)

    # Exact repeats: identical in every original column (including the pitch details and
    # monthly income) but with a different ID. These look like duplicated entries rather
    # than distinct customers; keeping them could place the same record in both train and
    # test, which would inflate the test score. Keep the first occurrence.
    content_cols = [c for c in df.columns if c not in ("Unnamed: 0", "CustomerID")]
    n_before = len(df)
    df = df.drop_duplicates(subset=content_cols, keep="first").reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} exact repeated records -> {len(df)} rows remain")

    dropped = NON_FEATURE_COLUMNS + POST_CONTACT_COLUMNS + CONTACT_DECISION_COLUMNS
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"prep.py: required columns missing: {missing}")
    print(f"Excluded from features: {dropped}")
    print(f"Model features ({len(FEATURES)}): {FEATURES}")

    X = df[FEATURES].copy()
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype("float64")
    y = df[TARGET].astype(int)

    # Stratify so both splits keep the same share of buyers (important when only ~1 in 5
    # customers is a positive case).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(args.splits_dir / "Xtrain.csv", index=False)
    X_test.to_csv(args.splits_dir / "Xtest.csv", index=False)
    y_train.to_frame(TARGET).to_csv(args.splits_dir / "ytrain.csv", index=False)
    y_test.to_frame(TARGET).to_csv(args.splits_dir / "ytest.csv", index=False)

    print("\nClass balance after the stratified split:")
    describe_split("train", y_train)
    describe_split("test", y_test)
    print(f"\nSaved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv to {args.splits_dir}/")


if __name__ == "__main__":
    main()
