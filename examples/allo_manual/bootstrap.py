"""Overlay the checked-in Python source with bindings built in this workspace."""
from pathlib import Path

here = Path(__file__).resolve().parent
source = here.parents[1] / "thirdparty/allo/allo"
overlay = here / "python/allo"
overlay.mkdir(parents=True, exist_ok=True)
for entry in source.iterdir():
    if entry.name in {"_mlir", "__pycache__"}:
        continue
    link = overlay / entry.name
    if not link.is_symlink():
        link.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
link = overlay / "_mlir"
if not link.is_symlink():
    link.symlink_to(here / "allo-build/tools/allo/_mlir", target_is_directory=True)
print(overlay)
