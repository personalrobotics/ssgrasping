"""Check ssgrasping's dependency boundary: importing it loads no simulator, planner,
viewer, behavior-tree, or robot package, and its distribution requires only ssrobot.

The one optional adapter, ``ssgrasping.sstsr``, is under the ``sstsr`` extra: the
distribution may require sstsr only under that extra, and importing the adapter may load
``tsr`` but nothing else forbidden. Without the extra installed, the adapter is reported
unavailable, not failed.

Each import runs in a fresh interpreter, so modules loaded by this script do not count.
With ``--installed``, it also requires ssgrasping to come from an installed wheel, not
this checkout, and checks that wheel's declared requirements.

    uv run python scripts/check_boundary.py [--installed] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import re
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

OPTIONAL = {"sstsr": ("ssgrasping.sstsr", ("sstsr", "numpy", "scipy"), ("tsr",))}
"""Each extra: the module it enables, the requirements it may add, and the forbidden
top-level modules that module alone may load."""

PROBE = """
import json, sys
import {module}
loaded = sorted({{name.split(".")[0] for name in sys.modules}})
print(json.dumps({{"loaded": loaded, "file": {module}.__file__}}))
"""


def probe(module: str) -> dict[str, object] | None:
    """The top-level modules a fresh interpreter loads importing ``module``, or None if
    it cannot be imported for want of a dependency."""
    run = subprocess.run(
        [sys.executable, "-P", "-c", PROBE.format(module=module)], capture_output=True, text=True
    )
    if run.returncode != 0:
        if "ModuleNotFoundError" in run.stderr:
            return None
        raise RuntimeError(f"importing {module} failed:\n{run.stderr}")
    found: dict[str, object] = json.loads(run.stdout)
    return found


_EXTRA = re.compile(r"""\bextra\s*==\s*['"]([^'"]+)['"]""")


def _marker_extra(requirement: str) -> str | None:
    """The extra a requirement's marker names, or None for an unconditional one."""
    if ";" not in requirement:
        return None
    found = _EXTRA.search(requirement.split(";", 1)[1])
    return found.group(1) if found else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--installed", action="store_true", help="require an installed wheel")
    parser.add_argument("--report", type=Path, help="write the findings as JSON")
    parser.add_argument(
        "--require-extra",
        action="append",
        default=[],
        choices=sorted(OPTIONAL),
        help="fail unless this extra's module imports (it is installed)",
    )
    args = parser.parse_args()

    found = probe("ssgrasping")
    if found is None:
        raise RuntimeError("ssgrasping cannot be imported")
    loaded = list(found["loaded"])  # type: ignore[call-overload]
    failures = [f"importing ssgrasping loads {name}" for name in FORBIDDEN if name in loaded]
    location = Path(str(found["file"])).resolve()
    from_checkout = location.is_relative_to(ROOT / "src")
    if args.installed and from_checkout:
        failures.append(f"ssgrasping was imported from the checkout: {location}")
    every = list(requires("ssgrasping") or [])
    declared = [r.split(";")[0].strip() for r in every if _marker_extra(r) is None]
    names = sorted({r.split("@")[0].split("[")[0].strip().lower() for r in declared})
    for name in names:
        if name not in ALLOWED_REQUIREMENTS:
            failures.append(f"the distribution requires {name}")
    extras = [
        r for r in declared if r.split("@")[0].strip() != r.split("@")[0].split("[")[0].strip()
    ]
    if extras:
        failures.append(f"the distribution requires ssrobot extras: {extras}")
    for requirement in every:
        extra = _marker_extra(requirement)
        if extra is None:
            continue
        name = requirement.split(";")[0].split("@")[0].split("[")[0]
        for c in "<>=!~ ":
            name = name.split(c)[0]
        if extra not in OPTIONAL or name.lower() not in OPTIONAL[extra][1]:
            failures.append(f"the {extra!r} extra requires {name}")

    adapters: dict[str, object] = {}
    for extra, (module, _, may_load) in OPTIONAL.items():
        adapter = probe(module)
        if adapter is None:
            adapters[extra] = "unavailable: the extra is not installed"
            if extra in args.require_extra:
                failures.append(f"{module} cannot be imported, but the {extra!r} extra is required")
            continue
        adapter_loaded = list(adapter["loaded"])  # type: ignore[call-overload]
        adapters[extra] = adapter_loaded
        failures += [
            f"importing {module} loads {name}"
            for name in FORBIDDEN
            if name in adapter_loaded and name not in may_load
        ]

    report = {
        "ssgrasping": str(location),
        "from_checkout": from_checkout,
        "requires": every,
        "loaded_top_level_modules": loaded,
        "optional_adapters": adapters,
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
