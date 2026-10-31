# Changelog

## Unreleased

Waits on the robot and the scene are on observed rest, not fixed durations
(personalrobotics/ssgrasping#35). ssrobot is pinned at 9a0195b, whose continuation gate
takes a rest watch's result.

### Migration

- `PlaceSpec.tare_wait_ns` is removed. The descent now starts once the arm is observed
  at rest, by the runtime's declared criteria; `PlaceSpec.rest_timeout_ns` (default
  `REST_TIMEOUT_NS`, 2 s) bounds how long that may take before `descend` fails.
- `LandingSpec.land_ns` is removed. `LandingSpec` requires `rest`, the
  `ssrobot.RestCriteria` by which the object counts as at rest, with their source, and
  `rest_timeout_ns`, the failure bound on observing it. There is no default for either.
  The landing is judged on the reading that established rest, and records it as
  `values["landed_rest"]`.
- `Lookahead.rest_wait_ns` is renamed `rest_timeout_ns`, and its default rises from
  500 ms to 2 s (`REST_TIMEOUT_NS`). A continuation now watches its arm's joints for
  rest instead of retrying ssrobot's gate each tick.
