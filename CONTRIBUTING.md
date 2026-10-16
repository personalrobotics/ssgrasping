# Contributing

## Development setup

```bash
git clone https://github.com/personalrobotics/ssgrasping
cd ssgrasping
uv sync
```

`uv sync` installs the development group: ssrobot with its MuJoCo and planning
integrations, and the Geodude and ADA robot packages at the commits ssrobot's
reference-robot gate pins. They are test fixtures, never runtime dependencies.

## Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv run python scripts/check_boundary.py
```

## Pull requests

- Branch from `main`, and open a PR with a clear summary and test plan.
- CI must pass. Review is by [@siddhss5](https://github.com/siddhss5), enforced through
  CODEOWNERS.

## Package manager

Use [uv](https://docs.astral.sh/uv/) only. The lockfile is committed and CI uses it as
is (`uv sync --locked`).

## License

By contributing, you agree that your contributions are licensed under the MIT License
(see [LICENSE](LICENSE)).
