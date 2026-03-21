from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def index_by_key(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    indexed: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        key = (row.get("product_key", ""), row.get("country", ""))
        if not key[0] or not key[1]:
            continue
        indexed[key] = row
    return indexed


def main() -> None:
    project_root = Path(__file__).resolve().parent

    siamese_path = project_root / "artifacts" / "siamese" / "sell_recommendations.csv"
    lightgbm_path = project_root / "artifacts" / "lightgbm" / "sell_recommendations.csv"
    output_path = project_root / "presentation.csv"

    if not siamese_path.exists() or not lightgbm_path.exists():
        raise FileNotFoundError(
            "Missing recommendation files. Run predict_siamese.py and predict_lightgbm.py first."
        )

    siamese_rows = read_csv(siamese_path)
    lightgbm_rows = read_csv(lightgbm_path)

    siamese_idx = index_by_key(siamese_rows)
    lightgbm_idx = index_by_key(lightgbm_rows)

    all_keys = sorted(set(siamese_idx.keys()) | set(lightgbm_idx.keys()))

    output_rows: List[Dict[str, object]] = []
    for key in all_keys:
        s = siamese_idx.get(key)
        l = lightgbm_idx.get(key)

        if s is None and l is None:
            continue

        current_sources = []
        if s is not None:
            current_sources.append(to_float(s.get("current_price_usd", "0")))
        if l is not None:
            current_sources.append(to_float(l.get("current_price_usd", "0")))
        current_price = sum(current_sources) / max(len(current_sources), 1)

        s_fair = to_float(s.get("estimated_fair_price_usd", "0")) if s is not None else 0.0
        l_fair = to_float(l.get("estimated_fair_price_usd", "0")) if l is not None else 0.0
        s_gap = to_float(s.get("price_gap_usd", "0")) if s is not None else 0.0
        l_gap = to_float(l.get("price_gap_usd", "0")) if l is not None else 0.0
        s_conf = to_float(s.get("confidence_score", "0")) if s is not None else 0.0
        l_conf = to_float(l.get("confidence_score", "0")) if l is not None else 0.0

        fair_values = []
        conf_values = []
        if s is not None:
            fair_values.append(s_fair)
            conf_values.append(s_conf)
        if l is not None:
            fair_values.append(l_fair)
            conf_values.append(l_conf)

        combined_fair = sum(fair_values) / max(len(fair_values), 1)
        combined_gap = current_price - combined_fair
        combined_conf = sum(conf_values) / max(len(conf_values), 1)

        if combined_gap > 5:
            combined_action = "lower_price"
        elif combined_gap < -5:
            combined_action = "raise_price"
        else:
            combined_action = "keep_price"

        siamese_action = s.get("action", "") if s is not None else ""
        lightgbm_action = l.get("action", "") if l is not None else ""
        if siamese_action and lightgbm_action:
            agreement = str(siamese_action == lightgbm_action).lower()
        else:
            agreement = "na"

        output_rows.append(
            {
                "product_key": key[0],
                "country": key[1],
                "current_price_usd": round(current_price, 4),
                "siamese_fair_price_usd": round(s_fair, 4) if s is not None else "",
                "lightgbm_fair_price_usd": round(l_fair, 4) if l is not None else "",
                "combined_fair_price_usd": round(combined_fair, 4),
                "siamese_gap_usd": round(s_gap, 4) if s is not None else "",
                "lightgbm_gap_usd": round(l_gap, 4) if l is not None else "",
                "combined_gap_usd": round(combined_gap, 4),
                "siamese_action": siamese_action,
                "lightgbm_action": lightgbm_action,
                "combined_action": combined_action,
                "siamese_confidence": round(s_conf, 4) if s is not None else "",
                "lightgbm_confidence": round(l_conf, 4) if l is not None else "",
                "combined_confidence": round(combined_conf, 4),
                "model_action_agreement": agreement,
            }
        )

    output_rows.sort(
        key=lambda r: abs(float(r["combined_gap_usd"])) * float(r["combined_confidence"]),
        reverse=True,
    )

    if not output_rows:
        raise RuntimeError("No overlapping product-country rows between models to export.")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    comparable = [r for r in output_rows if r["model_action_agreement"] in {"true", "false"}]
    agreement_count = sum(1 for r in comparable if r["model_action_agreement"] == "true")
    agreement_rate = agreement_count / max(len(comparable), 1)

    print(f"Rows in presentation.csv: {len(output_rows):,}")
    print(f"Comparable rows (both models present): {len(comparable):,}")
    print(f"Action agreement rows: {agreement_count:,} ({agreement_rate:.2%})")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
