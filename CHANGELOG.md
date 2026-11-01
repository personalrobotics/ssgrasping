# Changelog

## Unreleased

Skills estimate the manipulation transitions they pass through
(personalrobotics/ssgrasping#28), starting with `retained`.
- **Types.** `TransitionEstimate`, `TransitionEvaluation`, `Transition`,
  `TransitionOutcome`, `EstimateReason`, `Observed`, `Criterion`; `SkillResult.transitions`
  and `SkillResult.evaluations`; `Reason.TRANSITION_REFUTED`, whose cause is the
  `Transition`.
- **Retention.** `grasp`'s lift and the transfers of `drop` and `place` run under an
  ssrobot `OpeningGuard` (personalrobotics/ssrobot#227) midway between the opening the
  object is held at and the closing target. If it stops the carry, `retained` is
  refuted and the stage fails with the object reported not held.
- **Privileged state decides nothing.** `grasp`'s `retained` stage is the lift's
  estimate, and `drop`'s release no longer gates on the attachment monitor; the monitor
  is kept as ground truth in `SkillResult.evaluations`. `restrict` and `allow_target`
  still refuse to re-declare an attachment the monitor reports lost
  (personalrobotics/geodude#248).

### Compatibility

- `DropSpec` and `PlaceSpec` gain `close_opening` (default 0.0, `GraspSpec`'s default):
  the closing target the object is held with.
- `preflight` of `grasp`, `drop` and `place` requires the runtime to guard the arm on the
  opening channel; a runtime that cannot fails `preflight` with `unavailable`.
- A slip during a transfer now fails `transfer` with `transition_refuted` (cause
  `retained`) and `held` False, not `release` with `attachment_lost` and `held` True.
- ssrobot is pinned at the commit of personalrobotics/ssrobot#228.

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
