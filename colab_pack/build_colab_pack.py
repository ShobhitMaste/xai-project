#!/usr/bin/env python3
"""
Rebuild `colab_pack/` for a small Colab upload zip (no node_modules, no trained weights).

Run from `XAI/project`:
    python colab_pack/build_colab_pack.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent
ROOT = PACK.parent


def _rm(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _copy_backend() -> None:
    src = ROOT / "backend"
    dst = PACK / "backend"
    if not src.is_dir():
        raise SystemExit(f"Missing {src}")

    _rm(dst)

    def ignore(dirpath: str, names: list[str]) -> set[str]:
        out = set()
        base = Path(dirpath)
        rel = base.relative_to(src) if base != src else Path()
        rel_parts = rel.parts
        if rel_parts and rel_parts[0] == "model" and len(rel_parts) >= 2 and rel_parts[1] == "saved":
            return set(names)
        for n in names:
            if n in ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"):
                out.add(n)
            elif n.endswith(".pyc") or n.endswith(".pyo"):
                out.add(n)
        return out

    shutil.copytree(src, dst, ignore=ignore)
    saved = dst / "model" / "saved"
    if saved.exists():
        _rm(saved)
    saved.mkdir(parents=True, exist_ok=True)
    (saved / ".gitkeep").write_text(
        "# Empty on purpose; Colab training writes model.safetensors here.\n", encoding="utf-8"
    )


def _copy_resources() -> None:
    src = ROOT / "resources"
    dst = PACK / "resources"
    if not src.is_dir():
        raise SystemExit(f"Missing {src}")
    _rm(dst)

    def ignore(_, names: list[str]) -> set[str]:
        return {n for n in names if n == "__pycache__"}

    shutil.copytree(src, dst, ignore=ignore)


def _copy_frontend_light() -> None:
    src = ROOT / "frontend"
    dst = PACK / "frontend"
    if not src.is_dir():
        raise SystemExit(f"Missing {src}")
    _rm(dst)
    dst.mkdir(parents=True)
    for fname in ("package.json", "package-lock.json", ".env.example"):
        p = src / fname
        if p.is_file():
            shutil.copy2(p, dst / fname)
    for folder in ("public", "src"):
        sp = src / folder
        if sp.is_dir():
            shutil.copytree(
                sp,
                dst / folder,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )


def _copy_notebook() -> None:
    nb = ROOT / "Train_Stress_Detection_Colab.ipynb"
    if nb.is_file():
        shutil.copy2(nb, PACK / "Train_Stress_Detection_Colab.ipynb")


def _copy_optional_scripts() -> None:
    for name in ("colab_train_stress_all_in_one.py",):
        p = ROOT / name
        if p.is_file():
            shutil.copy2(p, PACK / name)


def main() -> None:
    if not ROOT.joinpath("backend", "requirements.txt").is_file():
        print("Run this script from inside the project (parent of colab_pack/).", file=sys.stderr)
        sys.exit(1)
    print("Building colab_pack from:", ROOT)
    _copy_backend()
    _copy_resources()
    _copy_frontend_light()
    _copy_notebook()
    _copy_optional_scripts()
    print("Done. Zip the folder:", PACK)
    print("  backend/   (model/saved is empty)")
    print("  frontend/  (source only, no node_modules)")
    print("  resources/")
    print("  Train_Stress_Detection_Colab.ipynb")


if __name__ == "__main__":
    main()
