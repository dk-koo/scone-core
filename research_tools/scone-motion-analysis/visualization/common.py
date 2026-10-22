"""Common plotting utilities for visualization modules."""

from pathlib import Path
from typing import Iterable, List


def save_figure(fig, output_base, formats: Iterable[str] = ("png", "svg"), **savefig_kwargs) -> List[str]:
    """Save a matplotlib figure to one or more formats and return saved paths."""
    base_path = Path(output_base)
    if base_path.suffix:
        base_path = base_path.with_suffix("")

    base_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file_format in formats:
        normalized_format = str(file_format).lower().lstrip(".")
        output_path = base_path.with_suffix(f".{normalized_format}")
        fig.savefig(str(output_path), format=normalized_format, **savefig_kwargs)
        saved_paths.append(str(output_path))

    return saved_paths
