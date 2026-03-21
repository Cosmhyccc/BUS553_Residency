from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class PriceRow:
    product_key: str
    country_code: str
    currency: str
    price_local: float
    price_usd: float
    category: str
    subcategory: str
    color_name: str
    gender_segment: str
    size_label: str


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


def build_product_key(row: Dict[str, str]) -> str:
    # The dataset has no product_id/model_number, so we use a stable composite key.
    fields = [
        row.get("product_name", ""),
        row.get("category", ""),
        row.get("subcategory", ""),
        row.get("color_name", ""),
        row.get("gender_segment", ""),
    ]
    return "||".join(str(f).strip().lower() for f in fields)


def load_price_rows(
    csv_path: Path,
    fx_to_usd: Dict[str, float],
    max_rows: Optional[int] = None,
) -> List[PriceRow]:
    rows: List[PriceRow] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if max_rows is not None and len(rows) >= max_rows:
                break

            currency = (raw.get("currency") or "").strip()
            if currency not in fx_to_usd:
                continue

            sale_price_local = parse_float(raw.get("sale_price_local"))
            regular_price_local = parse_float(raw.get("price_local"))
            # Use sale price when available to better reflect real market price.
            price_local = sale_price_local if sale_price_local is not None else regular_price_local
            if price_local is None:
                continue

            rows.append(
                PriceRow(
                    product_key=build_product_key(raw),
                    country_code=(raw.get("country_code") or "").strip(),
                    currency=currency,
                    price_local=price_local,
                    price_usd=price_local * fx_to_usd[currency],
                    category=(raw.get("category") or "").strip(),
                    subcategory=(raw.get("subcategory") or "").strip(),
                    color_name=(raw.get("color_name") or "").strip(),
                    gender_segment=(raw.get("gender_segment") or "").strip(),
                    size_label=(raw.get("size_label") or "").strip(),
                )
            )

    return rows


def build_pairs(rows: Iterable[PriceRow]) -> List[Tuple[PriceRow, PriceRow, float]]:
    """
    Build Siamese training pairs.
    USD price difference between  countries.
    """
    grouped: Dict[str, List[PriceRow]] = defaultdict(list)
    for row in rows:
        if row.product_key:
            grouped[row.product_key].append(row)

    pairs: List[Tuple[PriceRow, PriceRow, float]] = []
    for product_rows in grouped.values():
        if len(product_rows) < 2:
            continue

        for i in range(len(product_rows)):
            for j in range(i + 1, len(product_rows)):
                left = product_rows[i]
                right = product_rows[j]
                if left.country_code == right.country_code:
                    continue
                spread = left.price_usd - right.price_usd
                pairs.append((left, right, spread))

    return pairs
