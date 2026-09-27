"""
data_register.py - Register and validate the tourism dataset stored in this repository.

"Registering" here means: the dataset is version-controlled inside the GitHub repo at
tourism_project/data/tourism.csv, and this script records a fingerprint (SHA-256 hash),
row count, schema and target distribution of exactly that file. If the file does not
look like the dataset the pipeline expects, the script stops with a clear error so the
downstream jobs (data preparation and training) never run on bad input.

Run from the repository root:
    python tourism_project/model_building/data_register.py
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path("tourism_project/data/tourism.csv")
DEFAULT_REPORT_DIR = Path("tourism_project/outputs/registration")

TARGET = "ProdTaken"
ID_COLUMN = "CustomerID"
UNNAMED_INDEX = "Unnamed: 0"  # pandas' name for the blank first header in the CSV

# Every column the project relies on, grouped by the kind of values it must hold.
NUMERIC_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "CityTier", "DurationOfPitch",
    "NumberOfPersonVisiting", "NumberOfFollowups", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_COLUMNS = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]
REQUIRED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# Hard lower bound: far fewer rows than this means a truncated or wrong file.
MIN_ROWS = 1000
# Row count of the file supplied for this assignment (informational comparison only,
# so the pipeline can still retrain if the business later adds new rows).
DOCUMENTED_ROWS = 4128


def fail(message: str) -> None:
    """Print a clear error and stop with a non-zero exit code (fails the CI job)."""
    print(f"\nDATA REGISTRATION FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(df: pd.DataFrame) -> dict:
    """Run all checks. Hard failures call fail(); soft findings are returned as notes."""
    notes = []

    # 1. Required columns -------------------------------------------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        fail(f"required column(s) missing: {missing}. Found columns: {list(df.columns)}")
    unexpected = [c for c in df.columns if c not in REQUIRED_COLUMNS + [UNNAMED_INDEX]]
    if unexpected:
        notes.append(f"Unexpected extra columns (ignored downstream): {unexpected}")
    if UNNAMED_INDEX in df.columns:
        notes.append(f"'{UNNAMED_INDEX}' is a saved row index, not a feature; prep.py drops it.")

    # 2. Row count ----------------------------------------------------------------
    n_rows = len(df)
    if n_rows < MIN_ROWS:
        fail(f"only {n_rows} rows found; expected at least {MIN_ROWS}.")
    if n_rows != DOCUMENTED_ROWS:
        notes.append(f"Row count {n_rows} differs from the documented {DOCUMENTED_ROWS}.")

    # 3. Target values --------------------------------------------------------------
    if df[TARGET].isna().any():
        fail(f"target '{TARGET}' has {int(df[TARGET].isna().sum())} missing values.")
    target_values = set(pd.unique(df[TARGET]).tolist())
    if not target_values.issubset({0, 1}):
        fail(f"target '{TARGET}' must contain only 0/1, found {sorted(target_values)}.")
    if target_values != {0, 1}:
        fail(f"target '{TARGET}' must contain both classes, found only {sorted(target_values)}.")

    # 4. Duplicate customer IDs -------------------------------------------------------
    dup_ids = int(df[ID_COLUMN].duplicated().sum())
    if dup_ids > 0:
        fail(f"{dup_ids} duplicated {ID_COLUMN} values; each customer must appear once.")

    # 5. Basic data types ---------------------------------------------------------------
    bad_numeric = [c for c in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_numeric:
        fail(f"columns expected to be numeric are not: {bad_numeric}")
    bad_categorical = [
        c for c in CATEGORICAL_COLUMNS
        if not (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
    ]
    if bad_categorical:
        fail(f"columns expected to be text categories are not: {bad_categorical}")

    # 6. Soft findings: missing values and repeated records ----------------------------------
    missing_counts = {c: int(v) for c, v in df[REQUIRED_COLUMNS].isna().sum().items() if v > 0}
    content_cols = [c for c in df.columns if c not in (UNNAMED_INDEX, ID_COLUMN)]
    repeated_records = int(df.duplicated(subset=content_cols).sum())
    if repeated_records:
        notes.append(
            f"{repeated_records} rows repeat another row in every column except the "
            f"index/{ID_COLUMN}; prep.py decides how to handle them."
        )

    counts = df[TARGET].value_counts().sort_index()
    return {
        "rows": n_rows,
        "columns": int(df.shape[1]),
        "missing_values": missing_counts,
        "duplicate_customer_ids": dup_ids,
        "repeated_records_ignoring_ids": repeated_records,
        "target_counts": {str(k): int(v) for k, v in counts.items()},
        "target_positive_rate": round(float(df[TARGET].mean()), 4),
        "category_labels": {
            c: {str(k): int(v) for k, v in df[c].value_counts(dropna=False).items()}
            for c in CATEGORICAL_COLUMNS
        },
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Register and validate tourism.csv")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    if not args.data_path.exists():
        fail(f"dataset not found at {args.data_path}. It must be committed to the repository.")

    df = pd.read_csv(args.data_path)
    summary = validate(df)
    summary.update({
        "dataset_path": str(args.data_path),
        "sha256": sha256_of(args.data_path),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / "registration_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))

    # Concise human-readable summary (visible in the GitHub Actions log).
    print("=" * 60)
    print("DATASET REGISTERED")
    print("=" * 60)
    print(f"File           : {summary['dataset_path']}")
    print(f"SHA-256        : {summary['sha256'][:16]}...")
    print(f"Shape          : {summary['rows']} rows x {summary['columns']} columns")
    print(f"Target counts  : {summary['target_counts']} "
          f"(positive rate {summary['target_positive_rate']:.1%})")
    print(f"Missing values : {summary['missing_values'] or 'none'}")
    print(f"Duplicate IDs  : {summary['duplicate_customer_ids']}")
    for col in ("Gender", "MaritalStatus", "Occupation"):
        print(f"{col:<15}: {summary['category_labels'][col]}")
    for note in summary["notes"]:
        print(f"NOTE: {note}")
    print(f"Summary saved  : {report_path}")


if __name__ == "__main__":
    main()
