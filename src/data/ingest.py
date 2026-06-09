"""
Ingest Online Retail II dataset.

Usage:
    python -m src.data.ingest
    python -m src.data.ingest --file data/raw/my_custom_name.xlsx
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from src.utils.config import RAW_DIR, PROCESSED_DIR
from src.utils.logging_config import setup_logger

# ── Known sheet names across UCI and Kaggle versions ──────────────────────
SHEET_CANDIDATES = [
    ("Year 2009-2010", "Year 2010-2011"),   # UCI official
    ("2009-2010",      "2010-2011"),         # some mirrors
    (0, 1),                                  # positional fallback
]

# ── Known column renames (UCI v1 vs v2 naming) ────────────────────────────
COL_RENAMES = {
    "Customer ID": "CustomerID",            # Online Retail II (space variant)
    "UnitPrice":   "Price",                 # Online Retail I legacy name
}

OUTPUT_PATH = PROCESSED_DIR / "raw_combined.parquet"


def _find_xlsx(raw_dir: Path) -> Path:
    """Auto-detect the xlsx file in raw_dir."""
    xlsx_files = list(raw_dir.glob("*.xlsx"))
    if not xlsx_files:
        logger.error(
            f"No .xlsx found in {raw_dir}. "
            "Download Online Retail II from:\n"
            "  https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
            "Then drop the file into data/raw/"
        )
        sys.exit(1)
    if len(xlsx_files) > 1:
        logger.warning(f"Multiple xlsx found: {xlsx_files}. Using: {xlsx_files[0]}")
    return xlsx_files[0]


def _read_sheets(filepath: Path) -> pd.DataFrame:
    """Try known sheet name pairs; fall back to positional read."""
    for sheets in SHEET_CANDIDATES:
        try:
            dfs = []
            for sheet in sheets:
                df = pd.read_excel(
                    filepath,
                    sheet_name=sheet,
                    dtype={"CustomerID": str, "Customer ID": str},
                    engine="openpyxl",
                )
                dfs.append(df)
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded sheets {sheets} → {len(combined):,} rows")
            return combined
        except Exception:
            continue

    logger.error("Could not read any known sheet combination. Check xlsx structure.")
    sys.exit(1)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy/variant column names to canonical form."""
    df = df.rename(columns={k: v for k, v in COL_RENAMES.items() if k in df.columns})
    df.columns = [c.strip() for c in df.columns]
    return df


def load_raw(filepath: Path = None) -> pd.DataFrame:
    """
    Load Online Retail II xlsx → combined DataFrame.

    Parameters
    ----------
    filepath : Path, optional
        Path to xlsx. Auto-detected from data/raw/ if None.

    Returns
    -------
    pd.DataFrame
        Combined raw dataset with normalised column names.
    """
    if filepath is None:
        filepath = _find_xlsx(RAW_DIR)

    logger.info(f"Reading: {filepath}")
    df = _read_sheets(filepath)
    df = _normalise_columns(df)

    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Shape:   {df.shape}")
    logger.info(f"Memory:  {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return df


def save_parquet(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """Save DataFrame to parquet. Force object columns to string to avoid pyarrow type errors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Cast all object columns to str — prevents pyarrow mixed-type errors
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.success(f"Saved → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def run(filepath: Path = None) -> pd.DataFrame:
    """Full ingest step: load xlsx → save parquet → return df."""
    setup_logger()
    logger.info("=" * 50)
    logger.info("STEP 1/6 — DATA INGESTION")
    logger.info("=" * 50)

    df = load_raw(filepath)
    save_parquet(df)

    logger.success(f"Ingestion complete: {len(df):,} rows, {len(df.columns)} cols")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Online Retail II dataset")
    parser.add_argument("--file", type=Path, default=None,
                        help="Path to xlsx (auto-detected if omitted)")
    args = parser.parse_args()
    run(args.file)