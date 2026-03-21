from __future__ import annotations

import csv
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from models.lightgbm.config import (
    ARTIFACTS_DIR,
    DATA_FILE,
    FEATURE_FIELDS,
    FX_TO_USD,
    METADATA_PATH,
    MIN_RECOMMENDATION_GAP_USD,
    MODEL_PATH,
    SELL_ACTION_GAP_USD,
    TOP_K_OPPORTUNITIES,
    TOP_K_SELL_RECOMMENDATIONS,
)


@dataclass
class Row:
    product_key: str
    country_code: str
    price_usd: float
    features: Dict[str, str]


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def build_product_key(raw: Dict[str, str]) -> str:
    fields = [
        raw.get("product_name", ""),
        raw.get("category", ""),
        raw.get("subcategory", ""),
        raw.get("color_name", ""),
        raw.get("gender_segment", ""),
    ]
    return "||".join(str(x).strip().lower() for x in fields)


def load_rows(csv_path: Path, max_rows: Optional[int] = None) -> List[Row]:
    rows: List[Row] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if max_rows is not None and len(rows) >= max_rows:
                break

            currency = (raw.get("currency") or "").strip()
            if currency not in FX_TO_USD:
                continue

            sale_price_local = parse_float(raw.get("sale_price_local"))
            regular_price_local = parse_float(raw.get("price_local"))
            price_local = sale_price_local if sale_price_local is not None else regular_price_local
            if price_local is None:
                continue

            features = {field: (raw.get(field) or "").strip() for field in FEATURE_FIELDS}
            rows.append(
                Row(
                    product_key=build_product_key(raw),
                    country_code=(raw.get("country_code") or "").strip(),
                    price_usd=price_local * FX_TO_USD[currency],
                    features=features,
                )
            )
    return rows


def encode_rows(rows: List[Row], vocab: Dict[str, Dict[str, int]]) -> np.ndarray:
    x = np.zeros((len(rows), len(FEATURE_FIELDS)), dtype=np.int32)
    for i, row in enumerate(rows):
        for j, field in enumerate(FEATURE_FIELDS):
            x[i, j] = vocab[field].get(row.features[field], 0)
    return x


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        print(f"No rows to write for: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def action_from_gap(price_gap_usd: float) -> str:
    if price_gap_usd > SELL_ACTION_GAP_USD:
        return "lower_price"
    if price_gap_usd < -SELL_ACTION_GAP_USD:
        return "raise_price"
    return "keep_price"


def confidence_from_gap(abs_gap: float, val_mae: float) -> float:
    # Simple college-project confidence proxy based on error scale.
    scale = max(3.0 * val_mae, 1.0)
    return max(0.0, min(1.0, 1.0 - abs_gap / scale))


def main() -> None:
    project_root = Path(__file__).resolve().parent
    model_path = project_root / MODEL_PATH
    metadata_path = project_root / METADATA_PATH
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("LightGBM artifacts missing. Run train_lightgbm.py first.")

    with model_path.open("rb") as f:
        model = pickle.load(f)
    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)

    vocab = metadata["vocab"]
    val_mae = float(metadata["val_mae"])
    val_rmse = float(metadata["val_rmse"])

    rows = load_rows(csv_path=project_root / DATA_FILE, max_rows=180_000)
    x = encode_rows(rows, vocab)
    pred_fair = model.predict(x)

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    aggregate: Dict[tuple[str, str], Dict[str, float]] = {}
    for row, fair in zip(rows, pred_fair):
        current = float(row.price_usd)
        fair = float(fair)
        gap = current - fair
        abs_gap = abs(gap)
        if abs_gap < MIN_RECOMMENDATION_GAP_USD:
            continue

        confidence = confidence_from_gap(abs_gap=abs_gap, val_mae=val_mae)
        key = (row.product_key, row.country_code)
        if key not in aggregate:
            aggregate[key] = {
                "sum_current": 0.0,
                "sum_fair": 0.0,
                "sum_gap": 0.0,
                "sum_conf": 0.0,
                "count": 0.0,
            }
        aggregate[key]["sum_current"] += current
        aggregate[key]["sum_fair"] += fair
        aggregate[key]["sum_gap"] += gap
        aggregate[key]["sum_conf"] += confidence
        aggregate[key]["count"] += 1.0

    sell_rows: List[Dict[str, object]] = []
    for (product_key, country), agg in aggregate.items():
        n = max(agg["count"], 1.0)
        current_avg = agg["sum_current"] / n
        fair_avg = agg["sum_fair"] / n
        gap_avg = agg["sum_gap"] / n
        conf_avg = agg["sum_conf"] / n

        out_row = {
            "product_key": product_key,
            "country": country,
            "current_price_usd": round(current_avg, 4),
            "estimated_fair_price_usd": round(fair_avg, 4),
            "price_gap_usd": round(gap_avg, 4),
            "confidence_score": round(conf_avg, 4),
            "action": action_from_gap(gap_avg),
        }
        sell_rows.append(out_row)
        grouped[product_key].append(out_row)

    sell_rows.sort(
        key=lambda r: abs(float(r["price_gap_usd"])) * float(r["confidence_score"]),
        reverse=True,
    )
    top_sell = sell_rows[:TOP_K_SELL_RECOMMENDATIONS]

    opportunities: List[Dict[str, object]] = []
    for product_key, items in grouped.items():
        if len(items) < 2:
            continue
        cheapest = min(items, key=lambda r: float(r["price_gap_usd"]))
        priciest = max(items, key=lambda r: float(r["price_gap_usd"]))
        if cheapest["country"] == priciest["country"]:
            continue
        gap_spread = float(priciest["price_gap_usd"]) - float(cheapest["price_gap_usd"])
        opportunities.append(
            {
                "product_key": product_key,
                "underpriced_country": cheapest["country"],
                "underpriced_gap_usd": cheapest["price_gap_usd"],
                "overpriced_country": priciest["country"],
                "overpriced_gap_usd": priciest["price_gap_usd"],
                "gap_spread_usd": round(gap_spread, 4),
            }
        )

    opportunities.sort(key=lambda r: float(r["gap_spread_usd"]), reverse=True)
    top_opp = opportunities[:TOP_K_OPPORTUNITIES]

    sell_path = project_root / ARTIFACTS_DIR / "sell_recommendations.csv"
    opp_path = project_root / ARTIFACTS_DIR / "opportunities.csv"
    write_csv(sell_path, top_sell)
    write_csv(opp_path, top_opp)

    print(f"Rows scored: {len(rows):,}")
    print(f"Validation MAE used for confidence: {val_mae:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Sell recommendations found: {len(sell_rows):,}")
    print(f"Saved top {len(top_sell):,} sell recommendations to: {sell_path}")
    print(f"Opportunities found: {len(opportunities):,}")
    print(f"Saved top {len(top_opp):,} opportunities to: {opp_path}")


if __name__ == "__main__":
    main()
