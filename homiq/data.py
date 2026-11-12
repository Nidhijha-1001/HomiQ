"""Dataset download, cleaning, and exploration helpers for Indian housing data."""

import hashlib
import re
from pathlib import Path
from typing import Union
from urllib.request import urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Bengaluru_House_Data.csv"
TARGET = "price_inr"
DATA_SOURCE_URL = (
    "https://media.githubusercontent.com/media/roshan6192/Projects/"
    "cf909f5b61a4e566f4b7baf242d302ea0ce43b45/"
    "BREstate/Bengaluru_House_Data.csv"
)
DATA_SHA256 = "7b3e1361fcb64348d49eeb83d1e976738dc64bc978536df3438cf7b36f601b28"
LAKH_INR = 100_000
AREA_UNIT_TO_SQFT = {
    "sq meter": 10.7639104167,
    "sq yard": 9.0,
    "acre": 43_560.0,
    "cent": 435.6,
    "guntha": 1_089.0,
    "ground": 2_400.0,
    "perch": 272.25,
}


def _download_dataset(path: Path) -> None:
    """Download the pinned CC0 Kaggle data mirror and verify its contents."""
    with urlopen(DATA_SOURCE_URL, timeout=30) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(
            "The Bengaluru dataset download did not match its pinned SHA-256 "
            f"digest ({digest})."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _area_in_sqft(value: object) -> float:
    if pd.isna(value):
        return float("nan")

    text = str(value).strip().lower().replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        return float("nan")
    amount = float(numbers[0])
    if len(numbers) > 1 and re.search(r"\d\s*-\s*\d", text):
        amount = (amount + float(numbers[1])) / 2

    normalized_text = text.replace(".", "")
    for unit, multiplier in AREA_UNIT_TO_SQFT.items():
        if unit in normalized_text:
            return amount * multiplier
    return amount


def _bedroom_count(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:BHK|bedroom|bedrooms)\b",
        str(value),
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else float("nan")


def load_indian_housing(
    data_path: Union[Path, str] = DEFAULT_DATA_PATH, force_refresh: bool = False
) -> pd.DataFrame:
    """Load Bengaluru listings and normalize prices to rupees and area to sq ft."""
    path = Path(data_path)
    if force_refresh or not path.exists():
        _download_dataset(path)

    raw = pd.read_csv(path, low_memory=False)
    required = {"area_type", "location", "size", "total_sqft", "bath", "balcony", "price"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(
            "The Bengaluru housing dataset is missing required columns: "
            f"{missing}."
        )

    frame = pd.DataFrame(
        {
            "area_type": raw["area_type"].astype("string").str.strip(),
            "location": raw["location"].astype("string").str.strip(),
            "bhk": raw["size"].map(_bedroom_count),
            "total_sqft": raw["total_sqft"].map(_area_in_sqft),
            "bath": pd.to_numeric(raw["bath"], errors="coerce"),
            "balcony": pd.to_numeric(raw["balcony"], errors="coerce"),
            TARGET: pd.to_numeric(raw["price"], errors="coerce") * LAKH_INR,
        }
    )
    frame["location"] = frame["location"].str.replace(r"\s+", " ", regex=True)
    frame = frame.dropna(subset=["area_type", "location", "bhk", "total_sqft", TARGET])
    frame = frame[
        (frame["bhk"] > 0)
        & (frame["total_sqft"] > 0)
        & (frame[TARGET] > 0)
    ].reset_index(drop=True)
    return frame


def describe_dataset(frame: pd.DataFrame, usd_per_inr: float) -> None:
    """Print EDA summaries, converting the target for display only."""
    print(f"Dataset shape: {frame.shape}")
    print("\nColumns:")
    print(frame.columns.tolist())
    print("\nData types:")
    print(frame.dtypes.to_string())
    display_frame = frame.copy()
    if TARGET in display_frame:
        display_frame["ListedPriceUSD"] = display_frame[TARGET] * usd_per_inr
        display_frame = display_frame.drop(columns=[TARGET])
    print("\nNumeric summary (prices in USD):")
    print(display_frame.describe(include="number").T.to_string())
    print("\nMissing values (largest first):")
    missing = frame.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].to_string() or "No missing values.")
    print(f"\nExact duplicate rows: {frame.duplicated().sum()}")


def iqr_outlier_counts(frame: pd.DataFrame) -> pd.Series:
    """Count IQR-rule outliers per numeric column without removing observations."""
    numeric = frame.select_dtypes(include="number")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    spread = q3 - q1
    counts = ((numeric < q1 - 1.5 * spread) | (numeric > q3 + 1.5 * spread)).sum()
    return counts.sort_values(ascending=False)
