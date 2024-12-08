# Course AutoML ITMO: Regression with an Insurance Dataset

Author: Victor Barbarich

This repository contains the project submission for the **Light Auto ML course assignment**, focused
on solving a regression problem using an insurance dataset from the
[Kaggle Playground Series S4E12](https://www.kaggle.com/competitions/playground-series-s4e12/overview).

---

## Project Overview

The main objective of this project is to demonstrate mastery in solving machine learning problems
using two approaches:

1. **Baseline Solution**: Using Light Auto ML (LAMA) to quickly build a performant model.
2. **Custom Solution**: Developing an alternative approach without using LAMA, leveraging
   traditional machine learning techniques, pipelines, and feature engineering.

Key goals include:

- Building a structured and maintainable project following best practices.
- Surpassing the baseline performance with a custom model.
- Conducting thorough exploratory data analysis (EDA) and hypothesis-driven feature engineering.

---

## Repository Structure

```
├── LICENSE            <- Open-source license
├── Makefile           <- Commands for common project operations
├── README.md          <- Overview of the project (this file)
│
├── data
│   ├── external       <- Data from third-party sources (e.g., Kaggle datasets)
│   ├── interim        <- Intermediate, cleaned, and transformed data
│   ├── raw            <- Original dataset files (e.g., train.csv, test.csv)
│   └── processed      <- Final datasets ready for modeling
│
├── docs               <- Project documentation
│   ├── getting-started <- Installation and overview guides
│   ├── development     <- Contribution and coding guidelines
│   └── references      <- Information about datasets and models
│
├── models             <- Saved models, predictions, and summaries
│
├── notebooks          <- Jupyter notebooks for EDA and experimentation
│
├── pyproject.toml     <- Configuration file for project dependencies and tools
│
├── references         <- Supporting materials like data dictionaries or manuals
│
├── reports            <- Generated analysis and visualizations
│   └── figures        <- Graphics used in reporting
│
├── requirements.txt   <- Python dependencies for the project environment
│
├── setup.cfg          <- Configuration file for code style and linting
│
└── src                <- Source code for the project
    ├── __init__.py    <- Makes the src directory a Python module
    ├── config.py      <- Configuration settings
    ├── dataset.py     <- Data loading and preprocessing scripts
    ├── features.py    <- Feature engineering utilities
    ├── modeling
    │   ├── train.py   <- Training scripts for models
    │   ├── predict.py <- Model inference scripts
    │   └── __init__.py
    └── plots.py       <- Visualization utilities
```

---

## Key Components

### 1. **Exploratory Data Analysis (EDA)**

- Distribution of target variable and features
- Analysis of missing values, anomalies, and feature importance
- Correlation and dependency analysis
- Visualization of key insights

### 2. **Baseline Solution with LAMA**

- Implementation of a Light Auto ML pipeline
- Experimentation with multiple configurations
- Selection of the best-performing model for benchmarking

### 3. **Custom Solution**

- Development of a custom pipeline:
    - Preprocessing, handling missing values, feature engineering
    - Model selection and hyperparameter optimization
    - Ensemble methods to boost performance
- Comprehensive documentation of methodology and results

### 4. **Code Quality and Organization**

- Clean, modular, and well-documented code
- Adherence to best practices (PEP 8, SOLID principles)
- Use of tools like `scikit-learn` pipelines, logging, and exception handling

---

## Installation and Usage

### Prerequisites

- Python 3.12 [Installation Guide](https://www.python.org/downloads/)
- Poetry [Installation Guide](https://python-poetry.org/docs/#installation)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vktrbr/automl-itmo-insurance.git
   cd auto_ml_itmo_insurance_regression
   ```
2. Install dependencies:
   ```bash
   poetry install
   ```
3. Add `kaggle.json` to the `.kaggle` folder as `.kaggle/kaggle_example.json`
4. Running the Project using dvc
    ```bash
    dvc repro
    ```
5. If you have some problems with file locations, please run
    ```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    ```

---

## Results and Achievements

- **Baseline Performance**: Model trained using Light Auto ML achieved a strong benchmark result.
- **Custom Solution**: Improved upon the baseline with a tailored pipeline, yielding better accuracy
  and robustness.
- **Analysis**: Provided a detailed comparison of both approaches, highlighting their strengths and
  weaknesses.

---

## Contributions and Acknowledgments

This project is a learning exercise as part of the ITMO University course on Auto ML techniques.

For more information or to contribute, refer to
the [Contribution Guidelines](/docs/docs/development/contributing.md).
