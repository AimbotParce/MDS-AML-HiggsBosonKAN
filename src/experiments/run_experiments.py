import hashlib
import logging
import os
import pickle
import subprocess
import sys
from typing import List, Set

from .experiment_sets import Experiment, load_experiment_set

logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(
        self,
        experiments: Set[Experiment],
        no_cache: bool = False,
        cache_file: str = ".exp/cache/runs.pkl",
        no_python_executable_substitution: bool = False,
        extra_args: List[str] = [],
    ):
        self.experiments = experiments
        self.no_cache = no_cache
        self.cache_file = cache_file
        self.no_python_executable_substitution = no_python_executable_substitution
        self.extra_args = extra_args

        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def _run_command(self, cmd: List[str]):
        logger.info(f"Running command: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"  # Because MLflow prints uncode characters and Windows is dumb
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env
        )
        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
            logger.debug(f"Stdout: {result.stdout}")
            logger.debug(f"Stderr: {result.stderr}")
            raise RuntimeError(f"Experiment command failed: {' '.join(cmd)}")
        else:
            logger.info(f"Command succeeded")
            logger.debug(f"Stdout: {result.stdout}")
            logger.debug(f"Stderr: {result.stderr}")

    def run_all(self):
        # Load cache
        if os.path.exists(self.cache_file) and not self.no_cache:
            with open(self.cache_file, "rb") as f:
                cache = pickle.load(f)
        else:
            cache = set()

        for experiment in self.experiments:
            # Create a unique identifier for the experiment
            exp_id = hashlib.md5((experiment.executable + " ".join(sorted(experiment.arguments))).encode()).hexdigest()

            if exp_id in cache:
                logger.info(f"Skipping cached experiment: {experiment.executable} with args {experiment.arguments}")
                continue
            executable = experiment.executable
            if not self.no_python_executable_substitution and executable.startswith("python "):
                executable = executable.replace("python", sys.executable, 1)

            # The executable might contain spaces, so we split it properly. Same with args
            cmd = executable.split()
            for arg in experiment.arguments:
                if arg is not None:
                    cmd.extend(arg.split())
            cmd.extend(self.extra_args)

            self._run_command(cmd)

            # Update cache
            cache.add(exp_id)
            with open(self.cache_file, "wb") as f:
                pickle.dump(cache, f)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description=(
            "Run a set of experiments (as subprocesses) based on the provided configuration. "
            "You can pass additional arguments after '--' which will be forwarded to each experiment."
        )
    )
    parser.add_argument(
        "--experiments-config", type=str, required=True, help="Path to the experiment set configuration file."
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="If set, do not skip running experiments that have already been run."
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=".exp/cache/runs.pkl",
        help="File to use for caching experiment results.",
    )
    parser.add_argument(
        "--no-python-executable-substitution",
        action="store_true",
        help="If set, do not substitute 'python' executable in experiment commands.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase the verbosity of logging output (-v for print experiment outputs, -vv for DEBUG).",
    )
    args, extra_args = parser.parse_known_args()

    j = -1
    for j, extra_arg in enumerate(extra_args):
        if extra_arg == "--":
            break
        parser.error(f"Unknown argument: {extra_arg}")

    extra_args = extra_args[j + 1 :]

    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    experiment_set = load_experiment_set(args.experiments_config)
    runner = ExperimentRunner(
        experiments=experiment_set,
        no_cache=args.no_cache,
        cache_file=args.cache_file,
        no_python_executable_substitution=args.no_python_executable_substitution,
        extra_args=extra_args,
    )
    runner.run_all()
