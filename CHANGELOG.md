# Changelog

## Unreleased

A failed stage reports a stable, machine-readable reason
(personalrobotics/ssgrasping#34): `Stage.reason`, a new `Reason`, and `Stage.cause`, the
ssrobot code behind it. The README tabulates every reason. Each candidate a race tried
records each plan's `plan_status`, the planner's own status.

### Compatibility

- `Stage.reason` and `Stage.cause` are new fields at the end, defaulting to None, so
  existing construction of `Stage` is unchanged. Every stage name, detail, evidence kind,
  and held and attached state is as before.
- Applications that chose recovery or classified failures by parsing `Stage.detail`, or
  a race candidate's `detail`, should use the reason, the cause, and `plan_status`
  instead: detail is prose, and may be reworded in any release.

Waits on the robot are on observed rest, not fixed durations
(personalrobotics/ssgrasping#35). ssrobot is pinned at 9a0195b, whose continuation gate
takes a rest watch's result.

`landing` keeps its fixed `land_ns`. In MuJoCo, a can or bottle lying on a bin's floor
never comes to rest: under the native CCD collider, a cylinder at rest on a box of
comparable size creeps about 50 mm and back, about every 27 s
(google-deepmind/mujoco#3748, personalrobotics/prl_assets#27). Rolling resistance does
not stop it, so no rest criterion could pass for it. Landing moves to observed rest once
the collider is fixed (#36).

### Migration

- `PlaceSpec.tare_wait_ns` is removed. The descent now starts once the arm is observed
  at rest, by the runtime's declared criteria; `PlaceSpec.rest_timeout_ns` (default
  `REST_TIMEOUT_NS`, 2 s) bounds how long that may take before `descend` fails.
- `Lookahead.rest_wait_ns` is renamed `rest_timeout_ns`, and its default rises from
  500 ms to 2 s (`REST_TIMEOUT_NS`). A continuation now watches its arm's joints for
  rest instead of retrying ssrobot's gate each tick.
