import os
import pathlib


def add_gitignore(ckpt_path: os.PathLike | str) -> None:
    """
    Adds a .gitignore file to the models directory to ignore the specified checkpoint path.
    """

    ckpt_path = pathlib.Path(ckpt_path)

    models_dir = ckpt_path.parent
    os.makedirs(models_dir, exist_ok=True)
    ignore = "/" + ckpt_path.name
    gitignore = models_dir / ".gitignore"
    if gitignore.exists():
        prev_lines = gitignore.read_text().splitlines()
    else:
        prev_lines = []
    if ignore not in prev_lines:
        prev_lines.append(ignore)
        gitignore.write_text("\n".join(prev_lines) + "\n")
