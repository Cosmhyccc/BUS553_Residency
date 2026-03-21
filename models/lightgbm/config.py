from __future__ import annotations

from pathlib import Path

from models.siamese.config import FX_TO_USD


DATA_FILE = "nike.csv"

FEATURE_FIELDS = [
    "country_code",
    "category",
    "subcategory",
    "color_name",
    "gender_segment",
    "size_label",
    "brand_name",
    "sport_tags",
]

ARTIFACTS_DIR = Path("artifacts/lightgbm")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata.pkl"

TOP_K_SELL_RECOMMENDATIONS = 500
TOP_K_OPPORTUNITIES = 200

SELL_ACTION_GAP_USD = 5.0
MIN_RECOMMENDATION_GAP_USD = 5.0
