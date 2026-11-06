# Whether the planner's held geometry covers ADA's cylinder

This study answers the precondition #47 sets for ADA's place slice. While ADA carries
the cylinder, does the geometry the planner checks contain the cylinder the hand
actually holds? If it does not, a plan the validator passes can still put the real
cylinder into the table, the user, or the arm.

**It does not.** From the moment the cylinder is attached, the actual cylinder reaches
**35.7 mm** outside the planner's held volume (at most 38.2 mm, given the sampling), and
the clearance those motions plan with is 0. The ADA place slice is blocked on a
conservative held-object representation in ssrobot.

`ada.json` is the study's output. `scripts/study_held_geometry.py` produced it at
6cb1854, from a clean checkout, and two runs were byte-identical
(sha256 `6546e4ca0707aed79cc1692f13d42e6883ba264b022c3f44f5301df2d75264b4`).

## Method

ADA's bare hand (`ada_assets.models`) grasps the 4 cm by 12 cm, 100 g cylinder of
ssrobot's ADA reference scene, standing on the over-bed table. The run uses `grasp` on
registered object poses, with sstsr's `KinovaJaco2TwoFinger` candidates. It then
carries the cylinder as `place` would plan it: a transfer to the upright placement
14 cm away, raised by the 4 cm preplace distance, at 0.5 carry scaling, then a straight
3 cm descent that stops 1 cm short. `place` itself does not run, because on ADA it needs
an effort guard. The carry is unguarded.

Two departures from `grasp` as it stands, both what #47 would declare for the JACO2:

- **`reached_target` is not a miss.** The close ends `reached_target` (below), which
  `grasp` today takes for a miss, and on the re-close after the lift for a loss.
- **A 30 s gripper deadline,** as a bound on the close. The default 5 s ends the close
  `timed_out`.

**What the planner represents.** From attach, the planner checks:

- the hand's collision geoms at the measured joints: link 6, both fingers and
  fingertips, and the camera assembly, each mesh as its convex hull, as MuJoCo collides
  it;
- the cylinder at the attachment transform ssrobot stored at `attach`, read from the
  committed scene, never reconstructed.

**What is measured.** At every 2 ms tick, the study takes the largest signed distance
(negative inside) of any point of the actual cylinder's solid to that union, less the
validator's clearance. The actual cylinder is at its simulated pose: privileged
evaluation data that decides nothing in the run.

**Sampling bound.** The solid is sampled at 25,137 points on a cylindrical grid. The
signed distance to a union is 1-Lipschitz, and every point of the solid is within
2.56 mm of a sample, so the true largest protrusion lies within
[sampled, sampled + 2.56 mm]. The planned cylinder's distance is exact. Each hull's is
MuJoCo's `mj_geomDistance` from a probe sphere.

**When the cylinder has left the hand.** That is the first tick at which it touches no
hand geom, which is MuJoCo's contact condition, since every margin is zero. Ticks after
it are recorded as `<stage>:released`. The study's scene has no floor, so after the
release their distances are meaningless.

## Result

| | Position (m) | Orientation (wxyz) |
| --- | --- | --- |
| Registration, before the grasp | 0.3082, −0.6416, 0.7950 | 0.2707, 0, 0, 0.9627 |
| Planned cylinder at attach, from the stored transform | 0.3082, −0.6416, 0.7950 | 0.2707, 0, 0, 0.9627 |
| Live cylinder at attach (evaluation) | 0.3271, −0.6201, 0.7968 | 0.2742, 0.0788, 0.1877, 0.9399 |

- **The roll-in happens before attach.** The fingers first pinch the cylinder at
  opening 0.146, about 6.9 s into the run. Over the next 11 s they squeeze it into the
  palm and close round it, until the close ends `reached_target` at opening 0.010
  (18.30 s).
- **The stored transform keeps the pre-grasp pose.** On registered poses the stored
  transform is the registration expressed in the hand frame. So the planner holds the
  cylinder where it stood before the grasp, while the real one is already 28.7 mm away
  and tilted 0.41 rad.
- **The largest protrusion.** Through the lift, the re-close (`acquired`) and the first
  0.36 s of the transfer, the protrusion never falls below 33.0 mm in the 20 ms series
  (`series`), its least just before the release. The largest is
  35.67 mm, at 18.320 s in the lift, at the witness point (0.3544, −0.6106, 0.8518). The
  hand geoms cover none of it: its nearest member is the planned cylinder.

| Stage | Largest protrusion (mm) | At (s) | Ticks |
| --- | --- | --- | --- |
| `lift` | 35.67 | 18.320 | 122 |
| `acquired` | 34.99 | 18.564 | 1 |
| transfer, before the release | 35.14 | 18.606 | 160 |

**The release.** The cylinder leaves the hand at 18.926 s, 0.36 s into the transfer,
after its tilt has grown to 0.55 rad. Within 20 ms of the release the opening falls
from 0.0054 to 0.0013. That is past the midpoint a `place` transfer's opening guard
would use for this grasp, but it is a margin of a few thousandths of the opening.

## What this does not show

- **No transition ground truth.** It establishes no oracle for whether the cylinder was
  acquired or retained. It answers whether planning is conservative.
- **One scene, one grasp, no seeds.** The roll-in, and the release in the transfer,
  belong to this grasp depth on this cylinder in the simulated JACO2. Whether the
  physical hand behaves the same is not established.
- **The fix belongs in ssrobot.** A conservative held-object geometry covering the
  pinched-to-seated set, derived from declared hand and object geometry. Reattaching
  from the simulator's pose is not a fix.

Reproduce with `uv run python scripts/study_held_geometry.py OUT.json` (about 2 min).
