import os
from typing import Set

import pytest
import yaml

from src.experiments.experiment_sets import (
    ConfigCartesian,
    ConfigUnion,
    Experiment,
    ExperimentPart,
)


@pytest.fixture(scope="module")
def example_experiment_config_file() -> str:
    return os.path.join(os.path.dirname(__file__), "example_experiment_config.yaml")


@pytest.fixture(scope="module")
def expected_parsed_experiments() -> ConfigCartesian:
    return ConfigCartesian(
        cartesian=[
            ConfigUnion(
                union=[
                    ExperimentPart(executable="python notebooks/03-kan-experiment.py"),
                    ConfigCartesian(
                        cartesian=[
                            ExperimentPart(executable="python notebooks/04-mlp-experiment.py"),
                            ConfigUnion(
                                union=[
                                    ExperimentPart(argument="--l1-reg 0.01"),
                                    ExperimentPart(argument="--l1-reg 0.001"),
                                    ExperimentPart(argument=None),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
            ExperimentPart(argument="--cv-folds 5"),
            ConfigUnion(
                union=[
                    ExperimentPart(argument="--cv-fold-index 0"),
                    ExperimentPart(argument="--cv-fold-index 1"),
                    ExperimentPart(argument="--cv-fold-index 2"),
                    ExperimentPart(argument="--cv-fold-index 3"),
                    ExperimentPart(argument="--cv-fold-index 4"),
                ]
            ),
            ExperimentPart(argument="--epochs 30"),
            ConfigUnion(
                union=[
                    ExperimentPart(argument="--hidden-dim 2"),
                    ExperimentPart(argument="--hidden-dim 4"),
                    ExperimentPart(argument="--hidden-dim 8"),
                    ExperimentPart(argument="--hidden-dim 16"),
                    ExperimentPart(argument="--hidden-dim 32"),
                    ExperimentPart(argument="--hidden-dim 64"),
                    ExperimentPart(argument="--hidden-dim 128"),
                    ExperimentPart(argument="--hidden-dim 256"),
                    ExperimentPart(argument="--hidden-dim 512"),
                ]
            ),
        ]
    )


@pytest.fixture(scope="module")
def expected_compiled_experiments() -> Set[Experiment]:
    experiments: Set[Experiment] = set()
    cv_folds = 5
    epochs = 30
    hidden_dims = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    l1_regs = [None, 0.01, 0.001]
    cv_fold_indices = list(range(cv_folds))

    for hidden_dim in hidden_dims:
        for fold_index in cv_fold_indices:
            # KAN
            experiments.add(
                Experiment(
                    executable="python notebooks/03-kan-experiment.py",
                    arguments={
                        f"--cv-folds {cv_folds}",
                        f"--cv-fold-index {fold_index}",
                        f"--epochs {epochs}",
                        f"--hidden-dim {hidden_dim}",
                    },
                )
            )

            # MLP
            for l1_reg in l1_regs:
                args = {
                    f"--cv-folds {cv_folds}",
                    f"--cv-fold-index {fold_index}",
                    f"--epochs {epochs}",
                    f"--hidden-dim {hidden_dim}",
                }
                if l1_reg is not None:
                    args.add(f"--l1-reg {l1_reg}")

                experiments.add(
                    Experiment(
                        executable="python notebooks/04-mlp-experiment.py",
                        arguments=args,
                    )
                )
    return experiments


def test_experiment_parsing(example_experiment_config_file, expected_parsed_experiments):
    from src.experiments.experiment_sets import parse_experiment_config

    with open(example_experiment_config_file, "r") as f:
        config_content = yaml.safe_load(f)

    parsed_config = parse_experiment_config(config_content)
    assert parsed_config is not None
    assert parsed_config == expected_parsed_experiments


def test_experiment_compiling(expected_compiled_experiments, expected_parsed_experiments):
    from src.experiments.experiment_sets import compile_experiments

    experiments = compile_experiments(expected_parsed_experiments)
    assert experiments == expected_compiled_experiments
