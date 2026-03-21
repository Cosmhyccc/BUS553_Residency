from __future__ import annotations

import csv
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "LightGBM is required for train_lightgbm.py. Install with: pip install lightgbm"
    ) from exc

from models.lightgbm.config import (
    ARTIFACTS_DIR,
    DATA_FILE,
    FEATURE_FIELDS,
    FX_TO_USD,
    METADATA_PATH,
    MODEL_PATH,
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
    out: List[Row] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if max_rows is not None and len(out) >= max_rows:
                break

            currency = (raw.get("currency") or "").strip()
            if currency not in FX_TO_USD:
                continue

            sale_price_local = parse_float(raw.get("sale_price_local"))
            regular_price_local = parse_float(raw.get("price_local"))
            price_local = sale_price_local if sale_price_local is not None else regular_price_local
            if price_local is None:
                continue

            price_usd = price_local * FX_TO_USD[currency]
            features = {field: (raw.get(field) or "").strip() for field in FEATURE_FIELDS}

            out.append(
                Row(
                    product_key=build_product_key(raw),
                    country_code=(raw.get("country_code") or "").strip(),
                    price_usd=price_usd,
                    features=features,
                )
            )
    return out


def build_vocab(rows: List[Row]) -> Dict[str, Dict[str, int]]:
    vocab: Dict[str, Dict[str, int]] = {f: {"<UNK>": 0} for f in FEATURE_FIELDS}
    for row in rows:
        for field in FEATURE_FIELDS:
            value = row.features[field]
            if value not in vocab[field]:
                vocab[field][value] = len(vocab[field])
    return vocab


def encode_rows(rows: List[Row], vocab: Dict[str, Dict[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), len(FEATURE_FIELDS)), dtype=np.int32)
    y = np.zeros(len(rows), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, field in enumerate(FEATURE_FIELDS):
            x[i, j] = vocab[field].get(row.features[field], 0)
        y[i] = row.price_usd
    return x, y


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def main() -> None:
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / DATA_FILE
    artifacts_dir = project_root / ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading rows for LightGBM...")
    rows = load_rows(csv_path=csv_path, max_rows=180_000)
    print(f"Rows loaded: {len(rows):,}")
    if len(rows) < 1000:
        raise RuntimeError("Too few rows for meaningful LightGBM training.")

    vocab = build_vocab(rows)
    x_all, y_all = encode_rows(rows, vocab)

    idx = list(range(len(rows)))
    random.seed(42)
    random.shuffle(idx)
    split = int(0.85 * len(idx))
    train_idx = np.array(idx[:split], dtype=np.int32)
    val_idx = np.array(idx[split:], dtype=np.int32)

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l2",
        callbacks=[lgb.log_evaluation(period=100)],
    )

    y_val_pred = model.predict(x_val)
    val_mae = mae(y_val, y_val_pred)
    val_rmse = rmse(y_val, y_val_pred)
    print(f"Validation MAE (USD): {val_mae:.4f}")
    print(f"Validation RMSE (USD): {val_rmse:.4f}")

    with (project_root / MODEL_PATH).open("wb") as f:
        pickle.dump(model, f)
    with (project_root / METADATA_PATH).open("wb") as f:
        pickle.dump(
            {
                "vocab": vocab,
                "feature_fields": FEATURE_FIELDS,
                "fx_to_usd": FX_TO_USD,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            },
            f,
        )

    print(f"Saved model to: {project_root / MODEL_PATH}")
    print(f"Saved metadata to: {project_root / METADATA_PATH}")


if __name__ == "__main__":
    main()
