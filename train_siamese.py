from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "PyTorch is required for train_siamese.py. Install with: pip install torch"
    ) from exc

from models.siamese.data_prep import PriceRow, build_pairs, load_price_rows
from models.siamese.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    CAT_FIELDS,
    DATA_FILE,
    FX_TO_USD,
)
from models.siamese.model import SiameseConfig, SiamesePricingNetwork


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[PriceRow, PriceRow, float]],
        vocab: Dict[str, Dict[str, int]],
    ) -> None:
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode(self, row: PriceRow) -> Dict[str, int]:
        return {
            "country_code": self.vocab["country_code"].get(row.country_code, 0),
            "category": self.vocab["category"].get(row.category, 0),
            "subcategory": self.vocab["subcategory"].get(row.subcategory, 0),
            "color_name": self.vocab["color_name"].get(row.color_name, 0),
            "gender_segment": self.vocab["gender_segment"].get(row.gender_segment, 0),
            "size_label": self.vocab["size_label"].get(row.size_label, 0),
        }

    def __getitem__(self, idx: int):
        left, right, spread = self.pairs[idx]
        return {
            "left_cat": self._encode(left),
            "left_price": left.price_usd,
            "right_cat": self._encode(right),
            "right_price": right.price_usd,
            "target": spread,
        }


def build_vocab(rows: List[PriceRow]) -> Dict[str, Dict[str, int]]:
    vocab: Dict[str, Dict[str, int]] = {field: {"<UNK>": 0} for field in CAT_FIELDS}

    for row in rows:
        values = {
            "country_code": row.country_code,
            "category": row.category,
            "subcategory": row.subcategory,
            "color_name": row.color_name,
            "gender_segment": row.gender_segment,
            "size_label": row.size_label,
        }
        for field, value in values.items():
            if value not in vocab[field]:
                vocab[field][value] = len(vocab[field])

    return vocab


def collate_fn(batch):
    out = {
        "left_cat": {},
        "right_cat": {},
        "left_price": torch.tensor([item["left_price"] for item in batch], dtype=torch.float32),
        "right_price": torch.tensor([item["right_price"] for item in batch], dtype=torch.float32),
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.float32),
    }

    for field in CAT_FIELDS:
        out["left_cat"][field] = torch.tensor(
            [item["left_cat"][field] for item in batch], dtype=torch.long
        )
        out["right_cat"][field] = torch.tensor(
            [item["right_cat"][field] for item in batch], dtype=torch.long
        )

    return out


def fit_linear_calibration(
    predicted_abs: List[float], target_abs: List[float]
) -> Tuple[float, float]:
    """
    Fit y ~= slope * x + intercept using least squares.
    This calibrates predicted spread magnitudes to USD scale.
    """
    if not predicted_abs or not target_abs or len(predicted_abs) != len(target_abs):
        return 1.0, 0.0

    n = float(len(predicted_abs))
    sum_x = sum(predicted_abs)
    sum_y = sum(target_abs)
    sum_xx = sum(x * x for x in predicted_abs)
    sum_xy = sum(x * y for x, y in zip(predicted_abs, target_abs))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 1.0, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return float(slope), float(intercept)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / DATA_FILE
    artifacts_dir = project_root / ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading rows...")
    rows = load_price_rows(csv_path=csv_path, fx_to_usd=FX_TO_USD, max_rows=80_000)
    print(f"Loaded rows: {len(rows):,}")

    print("Building pairs...")
    pairs = build_pairs(rows)
    print(f"Built pairs: {len(pairs):,}")
    if not pairs:
        raise RuntimeError("No training pairs were generated.")

    random.shuffle(pairs)
    split = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    vocab = build_vocab(rows)
    cardinalities = {field: len(vocab[field]) for field in CAT_FIELDS}

    train_ds = PairDataset(train_pairs, vocab)
    val_ds = PairDataset(val_pairs, vocab)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = SiamesePricingNetwork(cardinalities=cardinalities, config=SiameseConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 3
    best_val_loss = float("inf")
    best_model_path = project_root / BEST_MODEL_PATH

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pred = model(
                left_cat=batch["left_cat"],
                left_price=batch["left_price"],
                right_cat=batch["right_cat"],
                right_price=batch["right_price"],
            )
            loss = loss_fn(pred, batch["target"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_pred_abs: List[float] = []
        val_target_abs: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    left_cat=batch["left_cat"],
                    left_price=batch["left_price"],
                    right_cat=batch["right_cat"],
                    right_price=batch["right_price"],
                )
                loss = loss_fn(pred, batch["target"])
                val_loss += loss.item()
                val_pred_abs.extend([abs(float(x)) for x in pred.detach().cpu().tolist()])
                val_target_abs.extend([abs(float(x)) for x in batch["target"].detach().cpu().tolist()])
        val_loss /= max(len(val_loader), 1)

        print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            cal_slope, cal_intercept = fit_linear_calibration(val_pred_abs, val_target_abs)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "cardinalities": cardinalities,
                    "cat_fields": CAT_FIELDS,
                    "fx_to_usd": FX_TO_USD,
                    "config": SiameseConfig().__dict__,
                    "best_val_loss": best_val_loss,
                    "calibration": {
                        "slope": cal_slope,
                        "intercept": cal_intercept,
                    },
                },
                best_model_path,
            )
            print(
                f"Saved new best checkpoint: {best_model_path} "
                f"(val_loss={best_val_loss:.4f}, cal_slope={cal_slope:.4f}, "
                f"cal_intercept={cal_intercept:.4f})"
            )


if __name__ == "__main__":
    main()
