"""Path handling for scripts executed from the solution directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical paths used by the pipeline."""

    solution_dir: Path
    data_dir: Path
    artifacts_dir: Path


def get_project_paths(anchor_file: str | Path) -> ProjectPaths:
    """Return canonical paths relative to a script or module file."""

    solution_dir = Path(anchor_file).resolve().parent
    if solution_dir.name == "src":
        solution_dir = solution_dir.parent

    return ProjectPaths(
        solution_dir=solution_dir,
        data_dir=solution_dir / "data",
        artifacts_dir=solution_dir / "artifacts",
    )
