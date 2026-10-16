"""Check ssgrasping's dependency boundary: importing it loads no simulator, planner,
viewer, behavior-tree, or robot package, and its distribution requires only ssrobot.

Each import runs in a fresh interpreter, so modules loaded by this script do not count.
With ``--installed``, it also requires ssgrasping to come from an installed wheel, not
this checkout, and checks that wheel's declared requirements.

    uv run python scripts/check_boundary.py [--installed] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import requires
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = (
    "mujoco",
    "sscbirrt",
    "ssik",
    "tsr",
    "geodude_assets",
    "ada_assets",
    "geodude",
    "ada_mj",
    "mj_manipulator",
    "mj_environment",
    "mj_viser",
    "viser",
    "py_trees",
)
"""Top-level modules ssgrasping must never import: simulators, planners, kinematics,
robot packages and applications, viewers, and behavior-tree frameworks."""

ALLOWED_REQUIREMENTS = ("ssrobot",)

PROBE = """
import json, sys
import ssgrasping
loaded = sorted({name.split(".")[0] for name in sys.modules})
print(json.dumps({"loaded": loaded, "file": ssgrasping.__file__}))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--installed", action="store_true", help="require an installed wheel")
    parser.add_argument("--report", type=Path, help="write the findings as JSON")
    args = parser.parse_args()

    probe = subprocess.run(
        [sys.executable, "-P", "-c", PROBE], capture_output=True, text=True, check=True
    )
    found = json.loads(probe.stdout)
    failures = [
        f"importing ssgrasping loads {name}" for name in FORBIDDEN if name in found["loaded"]
    ]
    location = Path(found["file"]).resolve()
    from_checkout = location.is_relative_to(ROOT / "src")
    if args.installed and from_checkout:
        failures.append(f"ssgrasping was imported from the checkout: {location}")
    declared = [r.split(";")[0].strip() for r in requires("ssgrasping") or []]
    names = sorted({r.split("@")[0].split("[")[0].strip().lower() for r in declared})
    for name in names:
        if name not in ALLOWED_REQUIREMENTS:
            failures.append(f"the distribution requires {name}")
    extras = [
        r for r in declared if r.split("@")[0].strip() != r.split("@")[0].split("[")[0].strip()
    ]
    if extras:
        failures.append(f"the distribution requires ssrobot extras: {extras}")

    report = {
        "ssgrasping": str(location),
        "from_checkout": from_checkout,
        "requires": declared,
        "loaded_top_level_modules": found["loaded"],
        "forbidden": list(FORBIDDEN),
        "failures": failures,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    for failure in failures:
        print(f"FAIL: {failure}")
    if not failures:
        print(f"OK: ssgrasping from {location} loads none of {', '.join(FORBIDDEN)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
