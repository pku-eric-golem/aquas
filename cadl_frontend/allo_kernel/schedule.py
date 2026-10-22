"""Turn the Allo code embedded in a CADL `allo` block into its schedule object."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .. import cadl_ast as ast


def load_schedule(kernel: ast.AlloKernel, directory: Path, cadl: Path):
    """Import the embedded Allo code and return its schedule object.

    Allo reads kernel source through `inspect`, so the code must live in a
    file. Leading blank lines keep its line numbers equal to the .cadl file.
    """
    module_path = directory / f"{kernel.name}_kernel.py"
    padding = "\n" * (kernel.source.span.line - 1)
    module_path.write_text(f"# From {cadl}" + padding + kernel.source.text + "\n")
    spec = importlib.util.spec_from_file_location(f"cadl_allo_{kernel.name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schedule = getattr(module, kernel.schedule, None)
    if schedule is None:
        raise ValueError(f"{cadl}: allo {kernel.name} names schedule '{kernel.schedule}', "
                         "which the embedded code does not define")
    if schedule.top_func_name != kernel.name:
        raise ValueError(f"{cadl}: schedule '{kernel.schedule}' customizes "
                         f"{schedule.top_func_name}, not {kernel.name}")
    return schedule
