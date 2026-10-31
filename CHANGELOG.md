# Changelog

## Unreleased

Waits on the robot are on observed rest, not fixed durations
(personalrobotics/ssgrasping#35). ssrobot is pinned at 9a0195b, whose continuation gate
takes a rest watch's result.

`landing` keeps its fixed `land_ns`. In MuJoCo, a can or bottle that lands on its side
in a bin rocks without decay, because nothing resists its rolling
(personalrobotics/prl_assets#27), so no rest criterion could pass for it. Landing moves
to observed rest once that is repaired (#36).

### Migration

- `PlaceSpec.tare_wait_ns` is removed. The descent now starts once the arm is observed
  at rest, by the runtime's declared criteria; `PlaceSpec.rest_timeout_ns` (default
  `REST_TIMEOUT_NS`, 2 s) bounds how long that may take before `descend` fails.
- `Lookahead.rest_wait_ns` is renamed `rest_timeout_ns`, and its default rises from
  500 ms to 2 s (`REST_TIMEOUT_NS`). A continuation now watches its arm's joints for
  rest instead of retrying ssrobot's gate each tick.
