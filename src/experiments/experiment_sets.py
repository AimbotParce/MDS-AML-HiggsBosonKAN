from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from pydantic import BaseModel, Field, model_validator


class ConfigUnion(BaseModel):
    union: "List[ConfigUnion | ConfigCartesian | ExperimentPart]"


class ConfigCartesian(BaseModel):
    cartesian: "List[ConfigUnion | ConfigCartesian | ExperimentPart]"


class ExperimentPart(BaseModel):
    argument: Optional[str] = Field(
        None, description="Argument string.", examples=["--config-path configs/experiment1.yaml"]
    )
    executable: Optional[str] = Field(
        None, description="Experiment executable.", examples=["python src/experiments/run_experiment.py"]
    )

    @model_validator(mode="before")
    def check_argument_or_executable_provided(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not "argument" in values and not "executable" in values:
            raise ValueError("At least one of 'argument' or 'executable' must be provided.")
        return values


class PartialExperiment(BaseModel):
    executable: Optional[str] = Field(
        None, description="Experiment executable.", examples=["python src/experiments/run_experiment.py"]
    )
    arguments: Set[str] = Field(
        default_factory=set,
        description="Set of argument strings to be passed to the executable.",
        examples=[{"--config-path configs/experiment1.yaml"}],
    )

    def __hash__(self):
        return hash((self.executable, frozenset(self.arguments)))

    def __eq__(self, other):
        if not isinstance(other, Experiment):
            return NotImplemented
        return self.executable == other.executable and self.arguments == other.arguments


class Experiment(BaseModel):
    executable: str = Field(
        ..., description="Experiment executable.", examples=["python src/experiments/run_experiment.py"]
    )
    arguments: Set[str] = Field(
        ...,
        description="Set of argument strings to be passed to the executable.",
        examples=[{"--config-path configs/experiment1.yaml"}],
    )

    def __hash__(self):
        return hash((self.executable, frozenset(self.arguments)))

    def __eq__(self, other):
        if not isinstance(other, Experiment):
            return NotImplemented
        return self.executable == other.executable and self.arguments == other.arguments


def parse_config_object(
    config_object: Dict[str, Any], breadcrumbs: List[str | int] = []
) -> ConfigUnion | ConfigCartesian | ExperimentPart:
    if config_object is None:
        return None
    elif not isinstance(config_object, dict):
        raise ParsingError("Config object must be either a dictionary or null.", breadcrumbs)
    keys = set(config_object.keys())
    if keys == {"$union"}:
        if not isinstance(config_object["$union"], list):
            raise ParsingError("The value of a $union must be a list.", breadcrumbs + ["$union"])
        return ConfigUnion(
            union=[
                parse_config_object(item, breadcrumbs + ["$union", j]) for j, item in enumerate(config_object["$union"])
            ]
        )
    elif keys == {"$cartesian"}:
        if not isinstance(config_object["$cartesian"], list):
            raise ParsingError("The value of a $cartesian must be a list.", breadcrumbs + ["$cartesian"])
        return ConfigCartesian(
            cartesian=[
                parse_config_object(item, breadcrumbs + ["$cartesian", j])
                for j, item in enumerate(config_object["$cartesian"])
            ]
        )
    elif keys.issubset({"executable", "argument"}):
        executable = config_object.get("executable")
        argument = config_object.get("argument")
        if executable is not None and not isinstance(executable, str):
            raise ParsingError("The value of 'executable' must be a string.", breadcrumbs + ["executable"])
        if argument is not None and not isinstance(argument, str):
            raise ParsingError("The value of 'argument' must be a string.", breadcrumbs + ["argument"])
        return ExperimentPart(argument=argument, executable=executable)
    else:
        raise ParsingError(
            f"Invalid config object. Must be either a $union, $cartesian, an ExperimentPart "
            f"with 'executable' and/or 'argument', or null. Found keys: {', '.join(keys)}",
            breadcrumbs,
        )


def parse_experiment_config(experiment_config: Dict | Any) -> ConfigUnion | ConfigCartesian:
    if isinstance(experiment_config, dict):
        if not "$union" in experiment_config and not "$cartesian" in experiment_config:
            raise ParsingError(
                "The top-level object in an experiment configuration must be either a $union or a $cartesian object.",
                [],
            )
        if "$union" in experiment_config and "$cartesian" in experiment_config:
            raise ParsingError(
                "The top-level object in an experiment configuration cannot contain both $union and $cartesian keys.",
                [],
            )
        res = parse_config_object(experiment_config)
        # This assert should always hold due to the checks above, but to satisfy the type checker we add it here.
        assert isinstance(
            res, (ConfigUnion, ConfigCartesian)
        ), "The top-level object in an experiment configuration must be a $union or a $cartesian object."
        return res
    else:
        raise ValueError(
            "The top-level object in an experiment configuration must be a dict with either a $union or a $cartesian key."
        )


def _merge_experiment_parts(
    exp1: PartialExperiment | ExperimentPart,
    exp2: PartialExperiment | ExperimentPart,
    breadcrumbs: List[str | int],
) -> PartialExperiment:
    # If both parts specify an executable, raise an error.
    if exp1.executable is not None and exp2.executable is not None:
        raise CompilationError("Found two executables in experiment definition.", breadcrumbs)
    executable = exp2.executable if exp2.executable is not None else exp1.executable

    # Merge arguments.
    arguments = set()
    if isinstance(exp1, ExperimentPart) and exp1.argument is not None:
        arguments.add(exp1.argument)
    elif isinstance(exp1, PartialExperiment):
        arguments.update(exp1.arguments)
    if isinstance(exp2, ExperimentPart) and exp2.argument is not None:
        arguments.add(exp2.argument)
    elif isinstance(exp2, PartialExperiment):
        arguments.update(exp2.arguments)
    return PartialExperiment(executable=executable, arguments=arguments)


def _compile_experiments_helper(
    experiment_config: ConfigUnion | ConfigCartesian | ExperimentPart, breadcrumbs: List[str | int] = []
) -> Set[PartialExperiment]:
    if isinstance(experiment_config, ExperimentPart):
        args = set()
        if experiment_config.argument is not None:
            args.add(experiment_config.argument)
        return {PartialExperiment(executable=experiment_config.executable, arguments=args)}
    elif isinstance(experiment_config, ConfigUnion):
        experiment_set: Set[PartialExperiment] = set()
        for i, sub_config in enumerate(experiment_config.union):
            sub_experiments = _compile_experiments_helper(sub_config, breadcrumbs + ["$union", i])
            experiment_set.update(sub_experiments)
        return experiment_set
    elif isinstance(experiment_config, ConfigCartesian):
        experiment_set: Set[PartialExperiment] = {PartialExperiment(executable=None, arguments=set())}
        for i, sub_config in enumerate(experiment_config.cartesian):
            sub_experiments = _compile_experiments_helper(sub_config, breadcrumbs + ["$cartesian", i])
            new_experiments: Set[PartialExperiment] = set()
            for existing_exp in experiment_set:
                for sub_exp in sub_experiments:
                    merged_exp = _merge_experiment_parts(existing_exp, sub_exp, breadcrumbs + ["$cartesian", i])
                    new_experiments.add(merged_exp)
            experiment_set = new_experiments
        return experiment_set
    else:
        raise CompilationError("Unknown experiment configuration type.", breadcrumbs)


def compile_experiments(experiment_config: ConfigUnion | ConfigCartesian) -> Set[Experiment]:
    experiment_set = _compile_experiments_helper(experiment_config)
    # Convert PartialExperiments to Experiments, ensuring all have executables.
    finalized_experiments: Set[Experiment] = set()
    for partial_exp in experiment_set:
        if partial_exp.executable is None:
            raise CompilationError("Experiment is missing an executable.", [])
        finalized_experiments.add(Experiment(executable=partial_exp.executable, arguments=partial_exp.arguments))
    return finalized_experiments


def load_experiment_set(config_path: str) -> Set[Experiment]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    parsed_config = parse_experiment_config(config)
    experiment_set = compile_experiments(parsed_config)
    return experiment_set


class ParsingError(Exception):
    @staticmethod
    def _breadcrumb_str(b: str | int) -> str:
        if isinstance(b, int):
            return f"[{b}]"
        return str(b)

    def __init__(self, message: str, breadcrumbs: List[str | int]):
        breadcrumb_str = ".".join(self._breadcrumb_str(b) for b in breadcrumbs)
        full_message = f"Error at {breadcrumb_str}: {message}"
        super().__init__(full_message)
        self.breadcrumbs = breadcrumbs


class CompilationError(Exception):
    @staticmethod
    def _breadcrumb_str(b: str | int) -> str:
        if isinstance(b, int):
            return f"[{b}]"
        return str(b)

    def __init__(self, message: str, breadcrumbs: List[str | int]):
        breadcrumb_str = ".".join(self._breadcrumb_str(b) for b in breadcrumbs)
        full_message = f"Error at {breadcrumb_str}: {message}"
        super().__init__(full_message)
        self.breadcrumbs = breadcrumbs


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Load and compile an experiment set configuration file.")
    parser.add_argument(
        "--experiment-set-config-path", type=str, required=True, help="Path to the experiment set configuration file."
    )
    args = parser.parse_args()
    with open(args.experiment_set_config_path, "r") as f:
        config = yaml.safe_load(f)
    parsed_config = parse_experiment_config(config)
    print(parsed_config)

    experiments = load_experiment_set(args.experiment_set_config_path)
    for experiment in experiments:
        print(f"Executable: {experiment.executable}")
        print(f"Arguments: {experiment.arguments}")
        print("-----")
