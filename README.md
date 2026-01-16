## Structure of this Repository

This repository contains the following main components.
- `data/`: This directory is intended to store datasets. It is structured into `raw/` for unprocessed data and `processed/` for cleaned and transformed data ready for analysis.
- `notebooks/`: This directory contains Jupyter notebooks used for data exploration, preprocessing, and model development.
- `src/`: This directory holds the source code for some helper modules developed during the project. It mainly contains three things:
  - `data/`: Functions for data loading and preprocessing.
  - `experiments/`: Module to compile and run experiment sets.
  - `trainers/`: Implementations of custom, analogous trainers (in torch) for MLPs and KANs.
  - ... Other utility functions.
- `figures/`: This directory is used to store generated plots and visualizations.
- `reports/`: This directory contains the project proposal and report document
- **Project Files**: These are files such as `README.md`, `pyproject.toml`, and `uv.lock`, which define dependencies and provide project documentation.
- **Data Version Control**: These are files related to DVC (Data Version Control) such as `dvc.yaml` and `.dvc/`. DVC is a tool that helps manage and version control large datasets and machine learning models, ensuring reproducibility and collaboration in data science projects.

Finally, the experiment results are stored in a common MLflow tracking server hosted at `http://mlflow.parcerisa.xyz/`. If you wish to explore the results, please contact the repository owner for access credentials.

## Installation

To install the necessary dependencies for this project, please follow the steps below:

1. **Clone the Repository**  
   Open your terminal and run the following command to clone the repository:
   ```bash
   git clone https://github.com/AimbotParce/MDS-AML-HiggsBosonATLAS.git
   ```

2. **Navigate to the Project Directory**
3. **Install Dependencies**
    **Recommended way:** using `uv`, which can be installed following instructions at https://docs.astral.sh/uv/getting-started/installation/
    ```bash
    uv sync
    ```
    **Alternative way:** using `pip` to install dependencies from `pyproject.toml` into a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install .
    ```
4. **Verify Installation**
    You can verify that the installation was successful by running:
    ```bash
    dvc --version
    ```
    If the command returns the version of DVC, the installation was successful. If you don't have DVC properly
    installed, you may run into issues in the next steps.

## Data Setup
To set up the data for this project, it's as easy as running the following command in the project directory:
```bash
dvc repro
```
This command will download the raw dataset into `data/raw/` and process it into `data/processed/` as required for the
project.

> [!NOTE]
> If you don't want to run the pipeline using dvc, you can do so manually using the following command (check that you have the virtual environment activated):
> ```bash
> python -c "import urllib.request; urllib.request.urlretrieve('https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz','data/raw/higgs-challenge.csv.gz')"
> ```
> And then running the data preprocessing script:
> ```bash
> python src/data/preprocess.py data/raw/higgs-challenge.csv.gz data/processed/
> ```

## Jupyter Notebooks and scripts

At this point, you should be able to run Jupyter notebooks in the `notebooks/` directory. You can execute them in order, as they have been named to reflect the intended sequence of execution:

1. `01-test-pykan.ipynb`: Initial tests and experiments with the PyKAN library.
2. `02-pykan-higgs-test.ipynb`: Testing PyKAN on the Higgs Boson dataset.
3. `03-kan-experiment.py`: Script to run KAN experiments.
4. `04-mlp-experiment.py`: Script to run MLP experiments.
5. `05-analysis.ipynb`: Analysis of the results obtained from the experiments, it connects to the MLflow server to fetch results, so make sure you have access to it.

## Report

The project report can be found in the root directory as a PDF file named `reports/HiggsBosonKAN.pdf`. In there, all the methodologies, experiments, results, and conclusions drawn from this project are detailed comprehensively.

