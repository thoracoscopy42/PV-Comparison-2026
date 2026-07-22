from pathlib import Path

from src.data import (
    chronological_split,
    load_csv,
    prepare_dataframe,
    print_missing_values,
    replace_infinite_values,
    save_splits,
    validate_time_index,
)


RAW_DATA_PATH = Path("data/raw/pv_data.csv")
PROCESSED_DATA_DIRECTORY = Path("data/processed")


def main() -> None:
    data = load_csv(RAW_DATA_PATH)
    data = prepare_dataframe(data)
    data = replace_infinite_values(data)

    print(f"Loaded rows:    {len(data)}")
    print(f"Loaded columns: {len(data.columns)}")

    validate_time_index(data)

    train_data, validation_data, test_data = chronological_split(data)

    print("\nChronological split:")
    print(f"Train:      {len(train_data)}")
    print(f"Validation: {len(validation_data)}")
    print(f"Test:       {len(test_data)}")

    print_missing_values("train", train_data)
    print_missing_values("validation", validation_data)
    print_missing_values("test", test_data)

    save_splits(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        output_directory=PROCESSED_DATA_DIRECTORY,
    )

    print(f"\nSaved processed splits in: {PROCESSED_DATA_DIRECTORY}")


if __name__ == "__main__":
    main()