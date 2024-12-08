"""
Preprocess the dataset

This script preprocesses the dataset by renaming the columns and saving it to a
feather file. Feather files are the fastest way to read and write data in
pandas.

Statistics about train.csv dataset:

- Read CSV file in 3.79 seconds
- Read feather file in 0.59 seconds
- CSV file size: 194100597 bytes ~ 194 MB
- Feather file size: 126564098 bytes ~ 126 MB


Study: [https://www.linkedin.com/pulse/comparative-study-among-csv-feather-pickle-parquet-loyola-gonz%C3%A1lez/](https://www.linkedin.com/pulse/comparative-study-among-csv-feather-pickle-parquet-loyola-gonz%C3%A1lez/)

"""

import os

import pandas as pd
import typer
from loguru import logger

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils import Timer

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    raw_data_dir: str = typer.Option(RAW_DATA_DIR, help="The raw data directory"),
    processed_data_dir: str = typer.Option(
        PROCESSED_DATA_DIR, help="The processed data directory"
    ),
):
    """
    Preprocess the dataset

    :param raw_data_dir: The raw data directory
    :param processed_data_dir: The processed data directory
    :return:
    """

    # Create the directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)

    # Preprocess the dataset
    for file in ["train.csv", "test.csv", "sample_submission.csv"]:
        dataset_path = os.path.join(raw_data_dir, file)
        target_path = os.path.join(processed_data_dir, file.replace(".csv", ".feather"))
        save_to_feather(dataset_path, target_path)

    logger.info("Dataset preprocessed")


def rename_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Replace spaces with underscores in the columns of the dataframe and
    convert them to uppercase

    :param dataset: The path to the dataset
    """

    old_columns = dataset.columns.tolist()
    new_columns = [col.replace(" ", "_").upper().strip() for col in old_columns]

    dataset.columns = new_columns

    logger.info(f"{old_columns=}")
    logger.info(f"{new_columns=}")

    return dataset


def save_to_feather(dataset_path: str, target_path: str):
    """
    Save the dataset to a feather file because it is the fastest way to read
    and write data in pandas

    :param dataset_path: The path to the dataset
    :param target_path: The path to save the feather file
    """
    if not target_path.endswith(".feather"):
        raise ValueError("The target path must end with .feather")

    csv_file_size = os.path.getsize(dataset_path)

    with Timer() as t:
        df = pd.read_csv(dataset_path)
    logger.info(f"Read CSV file in {t.elapsed:.2f} seconds")

    df = rename_columns(df)

    with Timer() as t:
        df.to_feather(target_path)
    logger.info(f"Saved feather file in {t.elapsed:.2f} seconds")

    logger.info(f"Dataset saved to {target_path}")
    logger.info(f"Dataset shape: {df.shape}")

    feather_file_size = os.path.getsize(target_path)
    with Timer() as t:
        _ = pd.read_feather(target_path)

    logger.info(f"Read feather file in {t.elapsed:.2f} seconds")

    logger.info(f"CSV file size: {csv_file_size} bytes")
    logger.info(f"Feather file size: {feather_file_size} bytes")

    return target_path


if __name__ == "__main__":
    app()
