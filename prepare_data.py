from pathlib import Path

from src.data import (
    chronological_split,
    load_csv,
    save_splits,
    validate_dataframe,
    prepare_structure,
)


RAW_DATA_PATH = Path("data/raw/pv_data.csv")
PROCESSED_DATA_DIRECTORY = Path("data/processed")


def main() -> None:
    data = load_csv(RAW_DATA_PATH)

    data = prepare_structure(data)

    print(data.shape)

    print(f"Loaded rows: {len(data)}")
    print(f"Loaded columns: {len(data.columns)}")

    validate_dataframe(data)

    train_data, validation_data, test_data = chronological_split(data)

    save_splits(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        output_directory=PROCESSED_DATA_DIRECTORY,
    )

    print()
    print("Chronological split:")
    print(f"Train:      {len(train_data)} rows")
    print(f"Validation: {len(validation_data)} rows")
    print(f"Test:       {len(test_data)} rows")
    print()
    print(f"Processed data saved in: {PROCESSED_DATA_DIRECTORY}")


if __name__ == "__main__":
    main()