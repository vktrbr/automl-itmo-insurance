"""
Load the dataset from Kaggle Regression with an Insurance Dataset

[
  https://www.kaggle.com/competitions/playground-series-s4e12/data
](https://www.kaggle.com/competitions/playground-series-s4e12/data)

Dataset Description
The dataset for this competition (both train and test) was generated from a
deep learning model trained on the Insurance Premium Prediction dataset.
Feature distributions are close to, but not exactly the same, as the original.
Feel free to use the original dataset as part of this competition,
both to explore differences and to see whether incorporating
the original in training improves model performance.


Files:

- train.csv - the training dataset; Premium Amount is the continuous target
- test.csv - the test dataset; your objective is to predict target
- sample_submission.csv - a sample submission file in the correct format

"""

import os
import zipfile

import typer
from loguru import logger

from src.config import EXTERNAL_DATA_DIR, KAGGLE_DATASET_NAME, RAW_DATA_DIR

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    destination: str = typer.Option(
        EXTERNAL_DATA_DIR, help="The destination folder to download the dataset"
    ),
    dataset: str = typer.Option(
        KAGGLE_DATASET_NAME, help="The dataset name to download"
    ),
    reload_if_exists: bool = typer.Option(
        False, help="If the dataset already exists, reload it"
    ),
):
    """
    Download the dataset from Kaggle

    ```bash
    dvc run -n download_dataset -d dataset.py -o data/raw python dataset.py
    ```

    :param destination: The destination folder
    :param dataset: The dataset name. We are using the Insurance Regression DS
    :param reload_if_exists: If the dataset already exists, reload it
    :return:
    """

    # Authenticate to Kaggle API
    authenticate()

    # Download the dataset
    path_to_dataset = get_data(destination, dataset, reload_if_exists)

    # Unzip the dataset
    files = unzip_data(path_to_dataset)

    return files


def authenticate():
    """
    Authenticate the user to Kaggle API
    """
    logger.info(os.environ.get("KAGGLE_CONFIG_DIR"))
    import kaggle.api

    # you should have a kaggle.json file in your home directory

    kaggle.api.authenticate()
    logger.info("Authenticated to Kaggle API")


def get_data(
    dataset_folder: str = EXTERNAL_DATA_DIR,
    dataset: str = KAGGLE_DATASET_NAME,
    reload_if_exists: bool = False,
) -> str:
    """
    Download the dataset from Kaggle

    :param dataset_folder: The destination folder
    :param dataset: The dataset name. We are using the Insurance Regression DS
    :param reload_if_exists: If the dataset already exists, reload it
    :return: The path to the downloaded dataset
    """

    os.makedirs(dataset_folder, exist_ok=True)

    destination = os.path.join(dataset_folder, f"{dataset}.zip")

    if not reload_if_exists and os.path.exists(destination):
        logger.info(f"Dataset already exists at {destination}")
        return destination

    import kaggle.api

    # Download the dataset
    logger.info(f"Downloading dataset {dataset} to {destination}")

    result = kaggle.api.competitions_data_download_files(
        dataset, _preload_content=False
    )

    with open(destination, "wb") as f:
        f.write(result.data)

    logger.success(f"SUCCESS: Downloaded dataset {dataset} to {destination}")

    return destination


def unzip_data(path_to_dataset: str, target_directory: str = RAW_DATA_DIR) -> list:
    """
    Unzip the dataset

    :param path_to_dataset:
    :param target_directory:
    :return:
    """

    logger.info(f"Unzipping dataset {path_to_dataset} to {target_directory}")

    with zipfile.ZipFile(path_to_dataset, "r") as zip_ref:
        zip_ref.extractall(target_directory)

    logger.success(f"SUCCESS: Unzipped dataset {path_to_dataset} to {target_directory}")

    files = os.listdir(target_directory)

    return files


if __name__ == "__main__":
    app()
