from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit("PyTorch is required for predict_siamese.py. Install with: pip install torch") from exc

from models.siamese.config import (
    BEST_MODEL_PATH,
    CAT_FIELDS,
    DATA_FILE,
    MIN_ACTUAL_SPREAD_USD,
    MIN_PREDICTED_SPREAD_USD,
    MIN_SELL_RECOMMENDATION_GAP_USD,
    SELL_ACTION_GAP_USD,
    TOP_K_OPPORTUNITIES,
    TOP_K_SELL_RECOMMENDATIONS,
)
from models.siamese.data_prep import PriceRow, load_price_rows
from models.siamese.model import SiameseConfig, SiamesePricingNetwork


def encode_row(row: PriceRow, vocab: Dict[str, Dict[str, int]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    encoded = {
        "country_code": vocab["country_code"].get(row.country_code, 0),
        "category": vocab["category"].get(row.category, 0),
        "subcategory": vocab["subcategory"].get(row.subcategory, 0),
        "color_name": vocab["color_name"].get(row.color_name, 0),
        "gender_segment": vocab["gender_segment"].get(row.gender_segment, 0),
        "size_label": vocab["size_label"].get(row.size_label, 0),
    }
    cat = {k: torch.tensor([v], dtype=torch.long) for k, v in encoded.items()}
    price = torch.tensor([row.price_usd], dtype=torch.float32)
    return cat, price


def predict_spread(
    model: SiamesePricingNetwork,
    left: PriceRow,
    right: PriceRow,
    vocab: Dict[str, Dict[str, int]],
) -> float:
    left_cat, left_price = encode_row(left, vocab)
    right_cat, right_price = encode_row(right, vocab)
    with torch.no_grad():
        pred = model(
            left_cat=left_cat,
            left_price=left_price,
            right_cat=right_cat,
            right_price=right_price,
        )
    return float(pred.item())


def apply_calibration(raw_abs_pred: float, slope: float, intercept: float) -> float:
    return max(0.0, slope * raw_abs_pred + intercept)


def compute_confidence(calibrated_pred_abs: float, actual_abs: float) -> float:
    """
    Confidence in [0, 1], higher when calibrated prediction
    agrees with observed spread magnitude.
    """
    base = max(calibrated_pred_abs, actual_abs, 1.0)
    error = abs(calibrated_pred_abs - actual_abs)
    return max(0.0, 1.0 - error / base)


def action_from_gap(price_gap_usd: float) -> str:
    """
    price_gap_usd = current_price - fair_price.
    """
    if price_gap_usd > SELL_ACTION_GAP_USD:
        return "lower_price"
    if price_gap_usd < -SELL_ACTION_GAP_USD:
        return "raise_price"
    return "keep_price"


def best_opportunity_for_product(
    model: SiamesePricingNetwork,
    rows: List[PriceRow],
    vocab: Dict[str, Dict[str, int]],
    cal_slope: float,
    cal_intercept: float,
) -> Dict[str, object] | None:
    if len(rows) < 2:
        return None

    best = None
    best_abs_pred = -1.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            left = rows[i]
            right = rows[j]
            if left.country_code == right.country_code:
                continue

            pred_spread = predict_spread(model, left, right, vocab)

            if pred_spread >= 0:
                sell_row = left
                buy_row = right
            else:
                sell_row = right
                buy_row = left

            raw_pred_abs = abs(pred_spread)
            calibrated_pred_abs = apply_calibration(raw_pred_abs, cal_slope, cal_intercept)
            actual_abs = abs(sell_row.price_usd - buy_row.price_usd)

            if calibrated_pred_abs < MIN_PREDICTED_SPREAD_USD:
                continue
            if actual_abs < MIN_ACTUAL_SPREAD_USD:
                continue

            confidence = compute_confidence(calibrated_pred_abs, actual_abs)

            if calibrated_pred_abs > best_abs_pred:
                best_abs_pred = calibrated_pred_abs
                best = {
                    "product_key": left.product_key,
                    "buy_country": buy_row.country_code,
                    "buy_price_usd": round(buy_row.price_usd, 4),
                    "sell_country": sell_row.country_code,
                    "sell_price_usd": round(sell_row.price_usd, 4),
                    "actual_spread_usd": round(actual_abs, 4),
                    "predicted_spread_usd_calibrated": round(calibrated_pred_abs, 4),
                    "confidence_score": round(confidence, 4),
                }

    return best


def sell_recommendations_for_product(
    model: SiamesePricingNetwork,
    rows: List[PriceRow],
    vocab: Dict[str, Dict[str, int]],
    cal_slope: float,
    cal_intercept: float,
) -> List[Dict[str, object]]:
    """
    Create sell-side recommendations using an anchor market.
    Anchor choice: lowest observed USD price for this product key.
    """
    if len(rows) < 2:
        return []

    anchor = min(rows, key=lambda r: r.price_usd)
    recommendations_by_country: Dict[str, Dict[str, object]] = {}

    for row in rows:
        if row.country_code == anchor.country_code and abs(row.price_usd - anchor.price_usd) < 1e-9:
            raw_pred_spread = 0.0
        else:
            # Signed spread: target_country - anchor_country
            raw_pred_spread = predict_spread(model, row, anchor, vocab)

        raw_abs = abs(raw_pred_spread)
        calibrated_abs = apply_calibration(raw_abs, cal_slope, cal_intercept)
        calibrated_signed = 0.0
        if raw_pred_spread != 0.0:
            calibrated_signed = calibrated_abs if raw_pred_spread > 0 else -calibrated_abs

        estimated_fair_price_usd = anchor.price_usd + calibrated_signed
        current_price_usd = row.price_usd
        price_gap_usd = current_price_usd - estimated_fair_price_usd

        actual_abs_vs_anchor = abs(current_price_usd - anchor.price_usd)
        confidence = compute_confidence(calibrated_abs, actual_abs_vs_anchor)
        action = action_from_gap(price_gap_usd)

        if abs(price_gap_usd) < MIN_SELL_RECOMMENDATION_GAP_USD:
            continue

        candidate = {
            "product_key": row.product_key,
            "country": row.country_code,
            "current_price_usd": round(current_price_usd, 4),
            "estimated_fair_price_usd": round(estimated_fair_price_usd, 4),
            "price_gap_usd": round(price_gap_usd, 4),
            "confidence_score": round(confidence, 4),
            "action": action,
            "anchor_country": anchor.country_code,
            "anchor_price_usd": round(anchor.price_usd, 4),
        }

        existing = recommendations_by_country.get(row.country_code)
        candidate_score = abs(float(candidate["price_gap_usd"])) * float(candidate["confidence_score"])
        if existing is None:
            recommendations_by_country[row.country_code] = candidate
            continue

        existing_score = abs(float(existing["price_gap_usd"])) * float(existing["confidence_score"])
        if candidate_score > existing_score:
            recommendations_by_country[row.country_code] = candidate

    return list(recommendations_by_country.values())


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No opportunities to write.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    checkpoint_path = project_root / BEST_MODEL_PATH

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run train_siamese.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = checkpoint["vocab"]
    fx_to_usd = checkpoint["fx_to_usd"]
    cardinalities = checkpoint["cardinalities"]
    config = SiameseConfig(**checkpoint["config"])
    calibration = checkpoint.get("calibration", {"slope": 1.0, "intercept": 0.0})
    cal_slope = float(calibration.get("slope", 1.0))
    cal_intercept = float(calibration.get("intercept", 0.0))

    model = SiamesePricingNetwork(cardinalities=cardinalities, config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = load_price_rows(
        csv_path=project_root / DATA_FILE,
        fx_to_usd=fx_to_usd,
        max_rows=120_000,
    )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row.product_key].append(row)

    opportunities: List[Dict[str, object]] = []
    sell_recommendations: List[Dict[str, object]] = []
    for product_rows in grouped.values():
        opp = best_opportunity_for_product(
            model=model,
            rows=product_rows,
            vocab=vocab,
            cal_slope=cal_slope,
            cal_intercept=cal_intercept,
        )
        if opp is not None:
            opportunities.append(opp)
        sell_recommendations.extend(
            sell_recommendations_for_product(
                model=model,
                rows=product_rows,
                vocab=vocab,
                cal_slope=cal_slope,
                cal_intercept=cal_intercept,
            )
        )

    opportunities.sort(
        key=lambda x: (
            float(x["predicted_spread_usd_calibrated"]),
            float(x["confidence_score"]),
        ),
        reverse=True,
    )
    top_k = opportunities[:TOP_K_OPPORTUNITIES]

    out_path = project_root / "artifacts" / "siamese" / "opportunities.csv"
    write_csv(out_path, top_k)

    sell_recommendations.sort(
        key=lambda x: abs(float(x["price_gap_usd"])) * float(x["confidence_score"]),
        reverse=True,
    )
    top_sell = sell_recommendations[:TOP_K_SELL_RECOMMENDATIONS]
    sell_out_path = project_root / "artifacts" / "siamese" / "sell_recommendations.csv"
    write_csv(sell_out_path, top_sell)

    print(f"Products evaluated: {len(grouped):,}")
    print(f"Opportunities found: {len(opportunities):,}")
    print(f"Saved top {len(top_k):,} opportunities to: {out_path}")
    print(f"Sell recommendations found: {len(sell_recommendations):,}")
    print(f"Saved top {len(top_sell):,} sell recommendations to: {sell_out_path}")
    print(f"Categorical fields used: {CAT_FIELDS}")
    print(f"Calibration used: slope={cal_slope:.4f}, intercept={cal_intercept:.4f}")


if __name__ == "__main__":
    main()
