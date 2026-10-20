# ssgrasping
Robot-neutral grasping and manipulation skills over ssrobot: pick, place, drop into a container, retract, guarded moves, and grasp checks

ssgrasping orchestrates ssrobot's public planning, execution, observation, and attachment
APIs into staged skills. It owns no simulator, planner, kinematics, retiming, or runtime
control: importing it loads none of them, and its wheel requires only ssrobot.

## Development

```bash
uv sync
uv run pytest
```

The development environment adds ssrobot's MuJoCo and planning integrations and the
Geodude and ADA robot packages, pinned to the commits ssrobot's reference-robot gate
uses. See [CONTRIBUTING.md](CONTRIBUTING.md) for every check CI runs.

