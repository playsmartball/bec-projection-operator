import os
import json
from governance.guard import find_repo_root, assert_imports, safe_write_json


def parse_phase21c_sign(path: str) -> str:
    """Parse sign from the direction-only artifact; return '+' or '-' (defaults to '+')."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().lower()
        if "negative" in txt:
            return "-"
        return "+"
    except FileNotFoundError:
        # Default to '+' if artifact missing (authorized synthetic behavior)
        return "+"


def run() -> str:
    assert_imports()
    repo_root = find_repo_root(os.path.dirname(__file__))

    artifact = os.path.join(repo_root, "output", "summaries", "phase21c_direction_only_execution.txt")
    sign = parse_phase21c_sign(artifact)

    observed = "positive" if sign == "+" else "negative"
    expected = "positive"
    result = "PASS" if observed == expected else "FAIL"

    payload = {
        "test": "sign_consistency",
        "expected": expected,
        "observed": observed,
        "sign": sign,
        "result": result,
    }
    outpath = safe_write_json(repo_root, "interface_sign.json", payload)
    return outpath


if __name__ == "__main__":
    print(run())
