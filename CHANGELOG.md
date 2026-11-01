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
  is kept as a privileged reference (`TransitionEvaluation.reference`, not ground
  truth) in `SkillResult.evaluations`. `restrict` and `allow_target`
  still refuse to re-declare an attachment the monitor reports lost
  (personalrobotics/geodude#248).
- **`placed` and `landed` are estimates, never stages** (personalrobotics/geodude#248).
  Both are recorded `unobserved` (`EstimateReason.NO_PORTABLE_SIGNAL`) at the confirmed
  release. Where the runtime has the object's privileged pose, `place` scores `placed`
  after its retreat, and `landing` scores the drop's `landed`, against the target
  regions, as evaluations only. The `retained` evaluation reads the runtime's violation
  report (`ctx.violation`) and records the object's pose at the carry's start and end.
- **Registered object poses.** The skills run in a context with ssrobot's
  `ObjectPoses.REGISTERED` (personalrobotics/ssrobot#230). `drop` and `place` take a
  `released` policy, called with the unobserved estimate once the release is confirmed,
  so the application can register or remove the object before anything else plans.
  `Candidates` takes `removals`, held objects a race's prediction expects released and
  then removed.

### Compatibility

- `DropSpec` and `PlaceSpec` gain `close_opening` (default 0.0, `GraspSpec`'s default):
  the closing target the object is held with.
- `preflight` of `grasp`, `drop` and `place` requires the runtime to guard the arm on the
  opening channel; a runtime that cannot fails `preflight` with `unavailable`.
- A slip during a transfer now fails `transfer` with `transition_refuted` (cause
  `retained`) and `held` False, not `release` with `attachment_lost` and `held` True.
- `place` has no `placed` stage, and ends after `retreat`; `values["placed_pose"]` is
  gone. `landing(ctx, spec, estimate)` takes the drop's `landed` estimate; its one stage
  is `observed`, and fails only if the object is still attached or has no privileged
  pose. `outside_regions` is now produced only by `place`'s `contact`.
- `grasp`'s `values["object_risen_m"]` is gone: the lift's evaluation holds the object's
  poses.
- ssrobot is pinned at the head of personalrobotics/ssrobot#232.

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
