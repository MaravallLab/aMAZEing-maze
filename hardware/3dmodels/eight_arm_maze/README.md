# Eight-arm maze

The configuration used for the auditory preference experiments: a corridor
running the width of the maze with four arms opening off each side, and an
entrance tunnel at the midpoint of one long side.

## Files

| File | What it is |
|---|---|
| `eight_arm_maze_modified.FCStd` | **The model of the rig.** Use this one |
| `eight_arm_maze_plan.pdf` / `.svg` / `.png` | The scale floor plan |
| `maze_layout.json` | The configuration, as squares on the 50 mm grid |
| `build_maze_model.py` | Regenerates the bare walls and posts from the layout |

`eight_arm_maze_modified.FCStd` is the accurate one. It holds the maze itself
and the rest of the rig around it: the table frame, the printed covers, the
transfer tunnel, the speaker with its grille, end caps and SpeakON connector,
the foam structure and board that carry it, the home cage with its wire lid and
water bottle, and the camera and infrared illuminator under the baseplate.
Parts carry their own materials, so the acrylic reads as acrylic.

The walls and posts inside it were not drawn by hand. They are generated from
`maze_layout.json`, which records which squares of the 50 mm post grid are
floor, which edges carry an acrylic panel and where the boundary is open. The
floor plan beside it comes from the same file, so the plan and the model cannot
disagree about the maze.

Everything else in the model was added by hand, because it is not something a
grid of squares can describe.

## Regenerating the walls and posts

Only needed if the maze layout itself changes. The script writes
`eight_arm_maze.FCStd`, a different name, so it cannot overwrite the model
above. You would then bring the new walls into the modified document, or rebuild
the surroundings on top of the new shell.

Inside FreeCAD, either add the script under `Macro > Macros...` and press
Execute, or paste this into the Python console under `View > Panels`:

```python
exec(open(r"hardware/3dmodels/eight_arm_maze/build_maze_model.py").read())
```

From a terminal, with no GUI:

```
FreeCADCmd build_maze_model.py
```

Without FreeCAD at all, the script still reports what it would build, which is
the quickest way to check a dimension or a part count:

```
python build_maze_model.py
```

## What it produces

```
layout    maze_layout.json: 21 floor squares, 37 panels, 8 detection regions
maze      350 x 300 mm, centred on a 700 x 700 mm plate
panels    100 x 50 x 3 mm acrylic (height x width x thickness)
posts     15 x 15 mm MakerBeam XL, 100 mm long
parts
     1  Baseplate
    37  Panels
    38  Posts
```

## Changing it

Where a change belongs depends on what it is.

**A change to the maze itself**, a wall moving, an arm changing length, the
tunnel getting longer: redraw the layout rather than editing any model. The
editor lives with the thesis figures (`figures/draw_maze_layout.py`): it shows
the post grid, you click the squares the mouse can walk on and the edges that
carry a panel, and it writes `maze_layout.json`. Copy that file here, redraw the
plan, and rerun the script. The thesis figure comes from the same file, so it
stays in step.

**A change to anything around the maze**, the speaker, the cage, the frame, the
ceiling, fillets, real T-slots in the posts: edit
`eight_arm_maze_modified.FCStd` directly. None of it can be derived from a grid
of squares.

To change a part, edit the constants at the top of `build_maze_model.py`. They
are the only dimensions in the model; everything else is derived.

## Where the dimensions come from

The parts are as the maze paper describes them: MakerBeam XL posts at
15 x 15 mm, acrylic panels 3 mm thick and 50 mm wide, on a baseplate drilled at
50 mm spacing. This configuration uses the taller 100 mm panels on 100 mm
posts. The four-arm tactile maze uses the 50 mm panels on 50 mm posts.

A panel is as wide as the post spacing, so in the model it overlaps each post
it meets by half a post. That is the panel seated in the T-slot, not a clash.

`docs/hardware/dimensions.md` disagrees: it records 10 x 10 mm posts, which is
the original MakerBeam rather than XL, and gives the panel width as both 40 mm
and 50 mm. The paper is taken as correct.

Two things nobody wrote down:

- **Where the maze sits on the baseplate.** Not recorded. The model centres it,
  which cannot be right, because the entrance tunnel has to reach the edge for
  the transport cage to dock. Measure this before making anything from the
  drawing.
- **Mounting hole diameter.** Not recorded. The bolts are M3, so the model uses
  3.4 mm clearance, and only if you ask: set `CUT_HOLES = True`.

## Checking it

The geometry is covered by `tests/test_maze_model.py`, which runs with the rest
of the suite and needs no FreeCAD. The FreeCAD half of the script is not
covered, because the test machine has no FreeCAD on it.
