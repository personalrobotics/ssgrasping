# Documentation figures

ssgrasping follows ssrobot's [documentation guide](https://github.com/personalrobotics/ssrobot/blob/main/docs/documentation.md)
and its [figure conventions](https://github.com/personalrobotics/ssrobot/blob/main/docs/figures/README.md):
plain, hand-written SVG that is both the editable source and the published artifact,
Palatino text, an SVG `title` and `desc` for accessibility, and the same color roles.
This file records the figures that belong to this repository.

| Figure | Owner | Question it answers |
| --- | --- | --- |
| `planning-ahead.svg` | `README.md` | Which motions of a pick and drop can be planned before the robot gets there? |

In `planning-ahead.svg`, amber marks motions, slate the outcomes that are observed
rather than planned, and violet planning done ahead of execution. Edit the SVG directly,
then parse it and render it at the README's width and at 360 pixels before committing:

```sh
xmllint --noout docs/figures/planning-ahead.svg
rsvg-convert -w 960 docs/figures/planning-ahead.svg -o /tmp/planning-ahead-960.png
rsvg-convert -w 360 docs/figures/planning-ahead.svg -o /tmp/planning-ahead-360.png
```
