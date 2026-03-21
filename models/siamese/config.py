from __future__ import annotations

from pathlib import Path


DATA_FILE = "nike.csv"

# Placeholder rates for scaffolding only.
# Replace with real FX rates before serious training.
FX_TO_USD = {
    "USD": 1.00,
    "EUR": 1.09,
    "GBP": 1.27,
    "JPY": 0.0067,
    "CNY": 0.14,
    "DKK": 0.15,
    "ILS": 0.27,
    "PLN": 0.25,
    "RON": 0.22,
    "SEK": 0.095,
}

CAT_FIELDS = [
    "country_code",
    "category",
    "subcategory",
    "color_name",
    "gender_segment",
    "size_label",
]

ARTIFACTS_DIR = Path("artifacts/siamese")
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pt"

# Business-facing filters for exported opportunities.
TOP_K_OPPORTUNITIES = 200
MIN_PREDICTED_SPREAD_USD = 5.0
MIN_ACTUAL_SPREAD_USD = 10.0

# Sell-side recommendation output controls.
TOP_K_SELL_RECOMMENDATIONS = 500
SELL_ACTION_GAP_USD = 5.0
MIN_SELL_RECOMMENDATION_GAP_USD = 5.0
