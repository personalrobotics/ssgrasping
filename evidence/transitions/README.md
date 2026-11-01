# What Geodude's portable signals observe of each manipulation transition

This study answers the first question of #28. Which of the signals a physical Geodude
reports decide each transition a manipulation skill passes through: `acquired`,
`retained`, `released`, `placed` and `landed`? The results shape `TransitionEstimate`
and the smallest runtime monitor the evidence requires. No skill changes here.

`summary.json` holds every run's truth and features, grouped by transition and case,
with the recording's provenance. `scripts/study_transitions.py` records and scores it.

## Method

Each run is `grasp`, then `place` or `drop`, on Geodude's right arm in MuJoCo, on a
manual clock. Every tick it records two things.

- **Portable signals:**
  - the gripper opening;
  - the wrist wrench, which is raw, untared, in the sensor's frame, and includes the
    hand's weight;
  - the arm's joint positions, velocities and efforts;
  - the hand pose.

  They are read with `portable_only=True`, so ssrobot refuses anything privileged.
- **Truth:**
  - the can's pose;
  - contact and normal force between the can and each finger's pad and follower;
  - the attachment monitor.

Truth is judged from physical state, never from a skill's verdict:

| Transition | True when |
| --- | --- |
| `acquired` | the can rose with the hand, by at least half the 0.06 m lift |
| `retained` | through the carry, the can stays within 0.01 m of where it was in the hand when the carry began (ssrobot's default attachment tolerance). Finger contact alone flickers while a heavy can shifts a few millimetres and stays held, so it is not the criterion. |
| `released` | at the end, the can is more than 0.05 m from the hand, touches neither finger, and is not held by the harness's weld |
| `placed`, `landed` | the can's final pose lies inside the place region, or the bin's interior, as the skills' own region checks judge it |

Each case runs over seeds 0–9, which perturb the can's start by up to 1 mm and 0.05 rad.
The can is the tests' 66 mm, 50 g cylinder. The `_heavy` cases give it 0.411 kg, the
mass of the heaviest recycling object in prl_assets (`cracker_box`).

| Case | What happens |
| --- | --- |
| `place`, `drop`, `place_heavy`, `drop_heavy` | a successful pick, then a place on the worktop or a drop into the bin |
| `missed` | the grasp aims 0.20 m from the can |
| `obstructed` | the same, with a fixed post of the can's size where the grasp aims |
| `lost` | a 20 kg can, which the grip cannot lift |
| `thrown` | the carry at the joints' full limits (`carry_scaling` 1.0) |
| `pulled`, `pulled_heavy` | the harness pulls the can down at 200 N for 100 ms, 0.5 s into the carry |
| `partial_open_drop`, `partial_open_place` | the release commands an opening of 0.5 |
| `stuck` | the harness welds the can to the right pad as the release begins, at their relative pose then |
| `outside_bin` | the drop zone is moved 0.35 m in +y, over the floor |
| `outside_region` | the place target's centre is moved 0.02 m past the worktop's edge |

The pull and the weld are faults no skill can cause. The harness makes them through the
runtime's MuJoCo data, in the script only.

## Findings

All values are from `summary.json`. A *gap* is the distance between the ranges of runs
where the transition truly happened and runs where it did not; a negative gap means the
ranges overlap. The wrist sensor's declared precision is the UR5e datasheet's 3.5 N, the
MJCF `noise` of `right_ur5e/ft_sensor_force`. MuJoCo does not apply it, so these
readings are noiseless.

### `acquired`: the gripper's terminal states decide a miss and a lost lift; nothing decides an obstruction

- **The close's terminal state refutes only a miss.**
  - `missed` reaches its target in 10 of 10 runs.
  - The close also stops short in all of `obstructed` (10) and `lost` (10).
- **The opening at the close cannot tell the can from a same-width post.**
  - The can reads 0.448–0.471, and the post 0.452–0.456.
  - A width-consistency check would catch only an obstruction of a different width.
- **A lost lift is decided by the lift's command or the re-close after it.**
  - In 7 of the 10 `lost` runs, the lift's own command fails (`goal_not_reached`): the
    20 kg can holds the arm back.
  - In the other 3, the lift completes and the re-close reaches its target.
  - The re-close stops short in all 120 runs that held the can.
- **The wrist load after the lift is biased by the arm's configuration.**
  - Lifting nothing reads −1.11 to −0.89 N (the 3 `lost` runs that completed their lift).
    The 50 g can reads −0.67 to −0.37 N (0.49 N expected), and the 0.411 kg can 2.66 to
    2.96 N (4.03 N expected).
  - The 50 g can's gap over nothing is 0.22 N, a sixteenth of the declared precision.
- **The obstruction case is decided only by the lift failing to plan,** because the
  fingers rest on the post: a planning refusal, not an observation.

### `retained`: the opening observes a slip, after it happens

- **The opening's fall during the carry separates cleanly.**
  - It is at most 0.007 in every retained run, and at least 0.448 in every lost run.
  - That 0.441 gap is over a hundred times the gripper's 1/255 resolution.
  - The fingers keep closing once the can is gone, because the simulated gripper holds
    its closing command. This post-slip response is the model's. It has not been
    confirmed on the physical 2F-140 (see Limitations).
- **The opening crosses half its fall 110–148 ms after a 50 g can slips 1 cm,** and
  142–404 ms after a 0.411 kg can does.
- **No real slip occurred.** A carry at full limits (`thrown`) kept the can in 10 of 10
  runs, so only the injected pull exercises a loss.

### `released`: the opening refutes a partial release but not a stuck object

- **A release that stops short is never a release.** It stopped short in all 19
  partial-open runs, and the can was still in hand in each.
- **Reaching the target does not confirm one.**
  - It reached its target in all 67 true releases.
  - It also reached its target in all 10 `stuck` runs, where the fingers opened around a
    can that stayed on the pad.
- **The wrist load after the release does tell a stuck can from a drop, but by less than
  the sensor's precision.**
  - A stuck can reads 0.38–0.54 N above the empty hand. A released one reads −0.43 to
    0.05 N.
  - The gap is 0.33 N, about a tenth of the declared precision.
- **A partly opened hand whose can rests on the worktop reads −0.19 to 0.24 N.** The
  support carries the can, so load cannot refute that case; the terminal state does.

### `placed` and `landed`: no portable signal sees them

- No recorded portable signal observes where the object rests after the release.
- Every `outside_bin` run (10) and every `outside_region` run whose place reached its
  release (8) confirmed the release first. A confirmed release therefore never implies
  `placed` or `landed`.

## What this means for #28

- **`acquired`** can be confirmed, or refuted as a miss or a lost lift, from the
  gripper's two terminal states and the lift's own command status. An obstruction by an object of the grasped width stays
  `unobserved` by these signals.
- **`retained`** has a strong portable signal in Geodude's MuJoCo model: the opening
  falling. Using it as a stop needs a runtime guard on the gripper opening during a carry.
  `WrenchGuard` watches only a wrench channel against an upper threshold, so it cannot
  express this. The result holds for the simulated hand until the physical 2F-140's
  post-slip response confirms it.
- **`released`**:
  - It is refuted by a release that stops short.
  - Confirming it needs the load's return. For objects whose weight is well below the
    sensor's precision, a release that reaches its target is `unobserved`, not
    confirmed.
  - A stuck object is therefore detectable only for heavy enough objects.
  - The estimate stops at `unobserved`. Whether to detach, retain or assume the release
    is the application's policy, and the application records which it chose (as #28 and
    Geodude#248 keep the estimate apart from the decision).
- **The wrist wrench is not usable as evidence of load without:**
  - an empty-hand model, as a function of the arm's configuration: the tool's mass and
    centre of mass, through forward kinematics;
  - each channel's declared precision, which `ChannelSpec` does not yet carry.
- **`placed` and `landed`** need a re-registered pose source. Without one they are
  `unobserved`, as #28 states.

## Limitations

- **The simulation is noiseless.** Margins are compared with the declared precision, but
  not measured under noise.
- **The scope is one object shape** (the 66 mm can, at two masses), on one robot and
  arm.
- **ADA is not studied.** Its published package holds the articutool in its fingers with
  finger–tool contact disabled, and its `tool_wrench` is the articutool's sensor, not a
  wrist's. Hand-object transitions on ADA wait for a hand-empty ADA (ssrobot#121).
- **Slips are injected,** as an external pull. A carry at full limits did not throw the
  can.
- **The slip's signature is the simulated gripper's.** The opening falls after a slip
  because MuJoCo's 2F-140 keeps driving toward its close target once the can is gone. The
  physical gripper reports the same opening signal, but its post-slip response has not
  been measured. Detecting a slip from the opening needs that hardware evidence.
- **Some seeds never reached the stage being judged:**

  | Case | Seeds | How the place ended |
  | --- | --- | --- |
  | `outside_region` | 2 and 8 | its transfer did not plan |
  | `place`, `partial_open_place` | 2 | `contact` failed: the can stopped outside the terminal band |

  These seeds count only where they reached the stage being judged.

## Reproduce

From a clean checkout of the commit `summary.json` names (`recorded.commit`), with
`OPENBLAS_CORETYPE=Haswell`:

```sh
uv run python scripts/study_transitions.py run OUT --jobs 32
uv run python scripts/study_transitions.py score OUT evidence/transitions/summary.json
```

`run` refuses an `OUT` that is not empty, and exits nonzero without writing
`provenance.json` if any recording fails. `score` refuses an `OUT` without
`provenance.json`, with any case and seed missing, repeated or unknown, or with a
recording whose recomputed sha256 differs from the one stored when it was recorded.

Each run is deterministic. Two complete runs of the grid on one host gave
byte-identical recordings, and each run's sha256 is in `summary.json`. The per-tick
recordings (`OUT/*.npz`) are regenerated, not kept.
