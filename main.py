from pathlib import Path
import csv


DATA_FILE = "nike.csv"


def preview_dataset(data_path: Path) -> None:
    """Print a small sanity-check preview of the dataset without external deps."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    row_count = 0
    column_names = []
    currencies = set()

    key_field_candidates = [
        "country_code",
        "currency",
        "price_local",
        "sale_price_local",
        "product_name",
        "product_id",
        "model_number",
        "category",
        "subcategory",
        "color",
        "gender",
        "size",
    ]
    missing_counts = {}

    with data_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        column_names = reader.fieldnames or []
        key_fields = [field for field in key_field_candidates if field in column_names]
        missing_counts = {field: 0 for field in key_fields}

        for row in reader:
            row_count += 1

            for field in key_fields:
                value = row.get(field)
                if value is None or str(value).strip() == "":
                    missing_counts[field] += 1

            currency_value = (row.get("currency") or "").strip()
            if currency_value:
                currencies.add(currency_value)

    print(f"Rows: {row_count:,}")
    print(f"Columns: {len(column_names):,}")
    print("\nAll columns:")
    for idx, name in enumerate(column_names, start=1):
        print(f"{idx:>2}. {name}")

    print("\nMissing values (key fields):")
    if not missing_counts:
        print("No expected key fields found in dataset header.")
    else:
        for field, missing_count in missing_counts.items():
            print(f"- {field}: {missing_count:,}")

    print("\nUnique currencies:")
    print(f"Count: {len(currencies):,}")
    print(sorted(currencies))


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / DATA_FILE

    print(f"Loading dataset from: {data_path}")
    preview_dataset(data_path)


if __name__ == "__main__":
    main()
