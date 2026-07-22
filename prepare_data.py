from pathlib import Path

from src.data import (
    add_target_availability,
    apply_imputer,
    chronological_split,
    fit_imputer,
    get_feature_columns,
    load_csv,
    prepare_dataframe,
    print_missing_values,
    replace_infinite_values,
    save_splits,
    validate_imputation,
    validate_time_index,
)


RAW_DATA_PATH = Path("data/raw/pv_data.csv")
PROCESSED_DATA_DIRECTORY = Path("data/processed")


def main() -> None:
    # Loading data
    data = load_csv(RAW_DATA_PATH)
    data = prepare_dataframe(data)
    data = replace_infinite_values(data)

    # Target def
    data = add_target_availability(data)

    print(f"Loaded rows:    {len(data)}")
    print(f"Loaded columns: {len(data.columns)}")

    # time validation
    validate_time_index(data)

    # chronological split
    train_data, validation_data, test_data = (
        chronological_split(data)
    )

    print("\nChronological split:")
    print(f"Train:      {len(train_data)}")
    print(f"Validation: {len(validation_data)}")
    print(f"Test:       {len(test_data)}")

    # report missing
    print_missing_values(
        "train before imputation",
        train_data,
    )

    print_missing_values(
        "validation before imputation",
        validation_data,
    )

    print_missing_values(
        "test before imputation",
        test_data,
    )

    # choose variables 
    feature_columns = get_feature_columns(
        train_data
    )

    print("\nFeature columns:")
    for column in feature_columns:
        print(f"- {column}")

    print(
        f"\nNumber of feature columns: "
        f"{len(feature_columns)}"
    )

    # compute median
    imputer = fit_imputer(
        train_data=train_data,
        feature_columns=feature_columns,
    )

    # fill data
    train_data = apply_imputer(
        data=train_data,
        feature_columns=feature_columns,
        imputer=imputer,
    )

    validation_data = apply_imputer(
        data=validation_data,
        feature_columns=feature_columns,
        imputer=imputer,
    )

    test_data = apply_imputer(
        data=test_data,
        feature_columns=feature_columns,
        imputer=imputer,
    )

    # last check
    validate_imputation(
        name="train",
        data=train_data,
        feature_columns=feature_columns,
    )

    validate_imputation(
        name="validation",
        data=validation_data,
        feature_columns=feature_columns,
    )

    validate_imputation(
        name="test",
        data=test_data,
        feature_columns=feature_columns,
    )

    # save splits
    save_splits(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        output_directory=PROCESSED_DATA_DIRECTORY,
    )

    print(
        f"\nSaved processed splits in: "
        f"{PROCESSED_DATA_DIRECTORY}"
    )


if __name__ == "__main__":
    main()