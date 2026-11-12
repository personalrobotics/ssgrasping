# Changelog

## Unreleased

A hand declares whether its gripper opening establishes a grasp
(`GraspingProfile.opening_establishes_grasp`, default True; personalrobotics/ssgrasping#47).
- **The declaration** is a keyword of `sstsr.profile`, and a field of `GraspSpec`,
  `PlaceSpec` and `DropSpec`; `GraspRegions.spec` copies the profile's.
- **When False,** as for the Kinova JACO2, whose fingers roll an object into the palm and
  can close to their target around it, the opening decides nothing: a close is not judged
  a miss, carries run with no opening guard, and `acquired` and `retained` are
  `unobserved` (`no_portable_signal`).
- **Proceeding on them is the application's separate policy,** `assume_unobserved` on
  `GraspSpec`, `PlaceSpec` and `DropSpec`, default False. Without it, the stage that
  needs the transition fails with the new reason `transition_unobserved` (cause: the
  transition), the object conservatively reported held. With it, the skill proceeds and
  records each transition it assumed in the new `SkillResult.assumed`.
- **The default is unchanged:** every existing test's evidence is identical but for the
  lookahead race's field documented as timing-dependent.
- **`tests/test_ada_grasp.py`:** ADA's bare hand on the 4 cm cylinder in open space.
  Without the declaration, the close reads as a miss; with it and no policy, the lift
  refuses `retained`; with the policy, the grasp and lift succeed, both transitions
  assumed.

ssrobot's pin moves to e1761b1, whose planning model leaves out geometry that cannot
collide, and with it sscbirrt to 3.16.0, which checks each trajectory piece at its own
speed (personalrobotics/ssrobot#251, personalrobotics/sscbirrt#279).

**Breaking: `clearance_m` is required** on `GraspSpec`, `PlaceSpec` and `DropSpec`, and on
`GraspRegions.spec`, with no default; the three specs are keyword-only. With the old
default of 0, a move through free space may skim what it passes by half a millimetre. In
the drop test on Linux, an upright carry planned that way dragged the can across the
worktop at the start of its transfer, and it turned 18 degrees in the hand. The tests
keep Geodude's 5 mm.

A drop is a placement released in the air (personalrobotics/ssgrasping#53).
- **Breaking: `DropSpec.regions` are for the object.**
  - They are where the object's frame may be when it is let go, such as sstsr's
    `OpeningPlacer` resting orientations over a container's opening.
  - `drop` composes them with the held attachment, as `place` does, through the shared
    `for_hand`.
  - Hand-frame drop zones must be given as object regions; composed with the grasp, the
    same hand poses come back exactly.
- **New required `DropSpec.fall_m`.** A transfer is kept only where its release is clear
  (`drop.release_clear`): the opened hand touches nothing but the object, and the
  object, released there and lowered `fall_m` every `FALL_STEP`, touches none of the
  robot's moving parts (`falls_clear`, naming each as `fall/<frame>`).
- **The race checks the drop that runs.** `Candidates`' `transfer` stage composes the
  regions with the hold the lift predicts. It rejects (ssrobot's `Stage.accept`) a
  transfer whose release is not clear, naming why, and tries the next seed.
  - The race plans with the fingers open, as they are before the grasp. That is more
    cautious than the closed hand `drop` carries: a release only the closed hand
    reaches fails the race as infeasible.
- **Where a drop let go.** `drop` records `values["released_at"]`: the object's pose by
  the measured joints and the attachment.
- **Not checked.** The object's turning as it falls, and its fall between steps.
- ssrobot's pin moves to 297e268, which adds `Stage.accept` (personalrobotics/ssrobot#250).

`obstructing` and `Ends` are public, so an application that commands a gripper itself,
outside a skill, checks it as the skills do (personalrobotics/geodude#272).
ssrobot's pin moves to 87dac58, whose reference gate pins ada_assets 0703586 as the tests
do.

Every gripper motion is checked against the scene before it runs
(personalrobotics/ssgrasping#54).
- **The check.** Before `Run.grip` commands a gripper, the planning scene is evaluated with
  the gripper's joints where the command leaves them: its declared `open_positions` or
  `closed_positions` (personalrobotics/ssrobot#243), at the arm's current configuration.
  A close and a release may touch the object and its support. The open before the
  pregrasp may touch nothing, and the close after the lift may touch the object. Any
  other contact fails the stage `gripper_obstructed` (new `Reason`), with cause `open` or
  `closed`, naming what the fingers would touch; nothing is commanded.
- **Releases chosen where the hand opens.** `drop`'s `choose` keeps a transfer only if
  the hand, opened where it ends, touches nothing but the object. `place`'s keeps one
  only if the hand, opened anywhere along the planned descent from `terminal_m` short of
  the regions to its end, touches nothing but the object and the target. A rejected seed
  is treated as an infeasible one, and `values["transfer_attempts"]` records what each
  seed's opened fingers would touch (`opened`, empty when clear). The place's band is
  sampled at most `BAND_STEP` apart in joint space, between waypoints too.
- **Declared positions are required.** `preflight` fails `unavailable` for a gripper that
  declares no `open_positions` and `closed_positions`, so no gripper is silently
  unchecked. Both reference robots declare them (geodude_assets e9538a2, ada_assets
  0703586).
- **`GraspSpec.lift_scaling`**, default 1.0: the fraction of the joints' velocity and
  acceleration limits the lift is planned at, in `grasp` and in `Candidates`' race.
- **Behaviour change.** A grasp whose fingers would close on something other than the
  object or its support is now refused at `close` (`gripper_obstructed`) instead of
  closing on it, attaching, and failing the lift (`test_grasp_place`'s `obstructed`).
- **Limitation.** Only where the fingers end is checked, not their sweep. Both reference
  hands' fingers move monotonically between open and closed, so an opening sweeps no
  further out than it ends. A close is checked fully closed, so in clutter a neighbour
  that fully closed fingertips would reach refuses it, though the object would stop
  them first. Contacts already present with the fingers where they are are not the
  command's, and are not counted. As for any plan, an object of unknown pose refuses
  the check. `Candidates`' race still checks only its transfer's feasibility. `drop`
  chooses its own transfer, with the check. Checking every gripper command in ssrobot
  itself is personalrobotics/ssrobot#249.
- Pins move to geodude_assets e9538a2 and ada_assets 0703586, and ssrobot's to 87dac58
  (below).

Candidates from several named, explicitly framed grasp regions
(personalrobotics/ssgrasping#46).
- **Regions.** A `GraspObject` lists `GraspRegion`s: a named primitive placed in the
  object frame by `T_object_region`, with its own strategies. Its single `primitive`
  field is replaced: an object of one shape is one region at the identity (unreleased
  API; the tests and README example migrate). `grasp_regions` selects regions by name and
  places each at `T_world_object @ T_object_region`, then the source's own frame
  convention, with no other shift.
- **Per-region outcomes.** `GraspRegions` holds a `RegionCandidates` per region, with
  its world frame and its own reason, so one region may be too wide for the hand while
  another has candidates; `tsrs`, `labels` (`region/strategy`), and `provenance` span
  them.
- **Provenance through the grasp.** `GraspSpec.pregrasp_labels` names each pregrasp
  region; `grasp` records in `values["pregrasp_region"]` the labels of the regions the
  pregrasp's planned path ends in, or none. `GraspSpec.pregrasps` is now a tuple.
- **Per-strategy apertures.** A strategy whose every span exceeds the hand's aperture is
  dropped (`RegionCandidates.dropped`) instead of passing because another strategy fits.

Grasp regions from explicit object geometry and a selected gripper profile
(personalrobotics/ssgrasping#3).
- **Types.** `Primitive` (cylinder, box, sphere, in a centred frame), `GraspObject`,
  `GraspingProfile`, `RegionTemplate`, `TemplateSource`, `GraspRegions`,
  `GeometryReason`, `UnsupportedPrimitive`, and `grasp_regions`, which backs the regions
  off by the grasp's own approach and records their provenance. Well-formed geometry that
  yields no region is a result with a reason, not an exception. `GraspSpec` is unchanged.
- **`ssgrasping.sstsr`** under the new `sstsr` extra: sstsr's hands as a
  `TemplateSource`, generating only the primitives sstsr's registry lists for a hand.
  `import ssgrasping` still loads no sstsr; `scripts/check_boundary.py` allows sstsr only
  under that extra and checks the adapter imports nothing else forbidden.
- **Fixed: regions rolled 180 degrees.** The matrix-to-quaternion conversion the tests and
  the README example used recovered each sign from the matrix's antisymmetric part, which
  vanishes at 180 degrees, so half of sstsr's side grasps became other poses. The adapter
  uses Shepperd's method.
- **Realistic test physics.** The test can is a full 12 fl oz can, 0.38 kg (it was 0.05
  kg). The 20 kg can that forced the grip to slip is replaced by a can stuck to its
  surface: with real masses no rigid can slips from the 2F-140, so the lift stalls and
  fails `command_failed`, with the can reported held. The descent's force margin and the
  carry's swing were re-measured at the new mass.
- **sstsr 4.2** in the test environment (4.1's closed-reach grasps, 4.2's JACO2 hand).
- **ADA cylinder candidates.** With ADA's bare-hand package and sstsr's
  `KinovaJaco2TwoFinger`, the same API makes cylinder candidates equal to sstsr's at the
  hand's palm grasp site, which the test checks is the palm sstsr frames the JACO2 in.
  Candidate generation only: no ADA grasp is executed. The test environment follows
  ssrobot's current reference-robot gate: geodude_assets f130a8f, and ada_assets 2d8f797
  (with the bare-hand package).
- **Runtime dependency: ssrobot 2d8285e to 1d3cb77**, which brings ssrobot's changes in
  between, including its effort guard (personalrobotics/ssrobot#237), to anyone installing
  ssgrasping. No ssgrasping source changed for it.

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
  truth) in `SkillResult.evaluations`
  (personalrobotics/geodude#248). `restrict` and `allow_target` change the attachment's
  allow set in place (`ctx.set_attachment_allowances`), consulting neither the runtime
  nor the monitor (personalrobotics/ssgrasping#44).
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
  `TransitionEvaluation.agrees` is None for an `unobserved` estimate.
  `Candidates` takes `removals`, held objects a race's prediction expects released and
  then removed.

### Removed (personalrobotics/ssgrasping#44)

- The planned-ahead hand-off between skills, which personalrobotics/ssgrasping#21
  decided not to adopt: `Follow`, `grasp(follow=)`, `Selection.grasp(follow=)`,
  `SkillExecution.upcoming`, `drop(planned=)` and `ssgrasping.drop.transfer_request`.
- The parked approach depth: `GraspSpec.deepest` and `approach_tries`; the approach is
  always `approach_m`.
- `Reason.ATTACHMENT_LOST`: no stage decides from the attachment monitor.
- Values no reader used: `closed_opening`, `lifted_opening`, `hand_allowance`,
  `contact_pose` and `descent`.
- `place`'s guarded descent runs through `ctx.execute`, as every other commanded stage.

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
