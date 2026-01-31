import os
import sys
import json
import hashlib
from typing import Optional

# Governance configuration
BANNED_MODULES = {
    "torch", "tensorflow", "jax", "jaxlib", "sklearn", "pandas", "matplotlib"
}


def find_repo_root(start_path: str) -> str:
    """Ascend from start_path until a directory containing .git is found."""
    d = os.path.abspath(start_path)
    for _ in range(10):
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    # Fallback: assume three levels up from windsurf/pipelines/rung17_interface
    return os.path.abspath(os.path.join(start_path, "..", "..", ".."))


def assert_imports() -> None:
    """Halt if any banned third-party modules are present in the runtime."""
    for name in list(sys.modules.keys()):
        top = name.split(".")[0]
        if top in BANNED_MODULES:
            raise RuntimeError(f"Governance violation: banned import detected: {top}")


def _safe_out_dir(repo_root: str) -> str:
    return os.path.realpath(os.path.join(repo_root, "output", "interface"))


def ensure_allowed_write(repo_root: str, target_path: str) -> None:
    outdir = _safe_out_dir(repo_root)
    real_target = os.path.realpath(target_path)
    if not (real_target == outdir or real_target.startswith(outdir + os.sep)):
        raise RuntimeError(f"Governance violation: write outside allowed dir: {real_target}")


def safe_write_json(repo_root: str, filename: str, payload: dict) -> str:
    outdir = _safe_out_dir(repo_root)
    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, filename)
    ensure_allowed_write(repo_root, fpath)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return fpath


def safe_write_text(repo_root: str, filename: str, text: str) -> str:
    outdir = _safe_out_dir(repo_root)
    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, filename)
    ensure_allowed_write(repo_root, fpath)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(text)
    return fpath


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

__all__ = [
    "find_repo_root",
    "assert_imports",
    "ensure_allowed_write",
    "safe_write_json",
    "safe_write_text",
    "sha256_file",
    "load_json",
]
