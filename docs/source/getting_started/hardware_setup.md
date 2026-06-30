# Hardware setup

This page is for building the physical maze rig — the parts, dimensions, and
where components were sourced — for anyone assembling the apparatus rather than
only running the software. It is the hardware counterpart to {doc}`installation`;
see {doc}`../user_guide/overview` for how the rig relates to the software. You do
not need to build the rig to run the analysis pipeline on already-recorded data —
only to collect new data.

The maze is built from laser-cut acrylic and 3D-printed components. All CAD
files live in `hardware/3dmodels/`, schematics in `hardware/drawings/`, and
assembly reference photos in `hardware/photos/`.

```{image} ../../../hardware/drawings/model.png
:alt: Maze with tuneable walls
:width: 480px
```

## Key components

- **Maze baseplate** with reconfigurable arm slots
- **Moveable walls** with a servo-driven cog mechanism
- **Camera holder** mounted under the maze
- **Electronics enclosure** for the Arduino and driver boards
- **Reward-delivery chute** with a servo-actuated gate triggered by IR
  detection of pellet delivery

## Dimensions

Based on the Rosenberg maze, adapted to MakerBeam construction.

| Part | Measurement |
|---|---|
| Baseplate | 700 × 700 mm, 10 mm thick (≥ ~7.5 mm works) |
| Acrylic wall panels | 50 × 50 mm, thickness set by the MakerBeam T-slot |
| MakerBeams | 10 × 10 mm profile, custom 50 mm lengths |
| Screws | M3 × 16 mm, with an M3 nut on the baseplate underside |
| Frame | 4 × 620 mm bars (perimeter) + 4 × 760 mm bars (legs) |

For a 4-arm maze you need roughly 50 side panels and 50 MakerBeam posts (with
spares), plus ceiling and floor plates and M3 bolts.

## Cameras and illumination

An IR camera (e.g. the ELP USB infrared webcam) paired with an IR illuminator
allows tracking in a dark, low-stress environment. Any OpenCV-compatible camera
works; set the device index via `video_input` in
[the experiment config](../user_guide/auditory_experiment.md).

## Where materials were sourced

- Acrylic: a local plastics supplier (Brighton & Hove Plastics)
- MakerBeam: the MakerBeam / MakerBeamXL shop
- Bolts: Screwfix (M3 pan-head machine screws)
- Frame box section: a general metals supplier

```{seealso}
The repository also ships two reference notes that this page distils:
`docs/dimensions.md` (full Rosenberg-maze conversion) and
`docs/local_materials_company.md` (supplier links).
```
