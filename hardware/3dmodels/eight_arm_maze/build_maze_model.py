#!/usr/bin/env python3
"""Build the eight-arm maze as a solid model, from the layout it was drawn as.

The maze is acrylic panels held between MakerBeam posts on a drilled baseplate,
so its geometry is a grid: posts sit at 50 mm spacing and every panel runs from
one post to the next. maze_layout.json records which squares are floor, which
edges carry a panel and where the boundary is open. Everything below follows
from that file and the part dimensions at the top of this one, so correcting a
part is a one-line change and correcting the layout is a change to the JSON.

Running it
----------
Inside FreeCAD, either

    Macro > Macros... > add this file > Execute

or, in View > Panels > Python console:

    exec(open(r"<path to this file>").read())

or from a terminal, without opening the GUI:

    FreeCADCmd build_maze_model.py

It writes, beside this file:

    eight_arm_maze.FCStd    the FreeCAD document
    eight_arm_maze.step     for any other CAD package
    eight_arm_maze.stl      for printing or for rendering

The geometry itself is worked out in plain Python and needs no FreeCAD, so

    python build_maze_model.py

prints the parts list and what it would build, and is the quickest way to check
a dimension before opening anything.

Coordinates
-----------
Millimetres throughout. X runs right and Y runs away from you in plan view, both
measured from the near left corner of the baseplate; Z is up from the underside
of the baseplate. The layout file counts rows downwards from the top, which is
how you read a plan, so the row index is flipped on the way in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYOUT = HERE / "maze_layout.json"

# -- parts ------------------------------------------------------------------
# These are the parts as the maze paper describes them. Where the paper and
# docs/hardware/dimensions.md disagree, the paper wins and the difference is
# noted.

CELL_MM = 50.0            # post spacing, and the baseplate hole pitch
POST_MM = 15.0            # MakerBeam XL, 15 x 15 mm square section.
                          # dimensions.md says 10 x 10 mm, which is the
                          # original MakerBeam rather than XL.
POST_LENGTH_MM = 100.0    # posts come 50 or 100 mm long. The 100 mm ones are
                          # what carries a 100 mm panel.
PANEL_WIDTH_MM = 50.0     # acrylic panel, width. It spans one post pitch, so
                          # it seats into the T-slot of the post at each end
                          # and overlaps each post by half a post.
PANEL_HEIGHT_MM = 100.0   # the eight-arm auditory maze used the taller
                          # panels. The four-arm tactile maze uses the 50 mm
                          # ones, on 50 mm posts.
PANEL_THICK_MM = 3.0

PLATE_MM = 700.0          # baseplate. The paper gives 700 x 700 x 10 mm for
                          # the four-arm version; the eight-arm plate is not
                          # recorded, and this maze fits well inside it.
PLATE_THICK_MM = 10.0
HOLE_PITCH_MM = 50.0
HOLE_DIA_MM = 3.4         # M3 clearance. The drilled size is not recorded.

# Cutting the full grid of mounting holes is slower and asserts a drill size
# that nobody wrote down, so it is off unless asked for.
CUT_HOLES = False


@dataclass(frozen=True)
class Box:
    """One rectangular solid, as its near-left-bottom corner and its size."""
    name: str
    group: str
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float


def norm(seg):
    """A wall segment in a fixed order, so one panel is only ever one key."""
    a, b = tuple(seg[0]), tuple(seg[1])
    return (a, b) if a <= b else (b, a)


def boundary_walls(cells):
    """Every edge with open ground on one side of it and not the other."""
    segs = set()
    for (c, r) in cells:
        if (c, r - 1) not in cells:
            segs.add(norm(((c, r), (c + 1, r))))
        if (c, r + 1) not in cells:
            segs.add(norm(((c, r + 1), (c + 1, r + 1))))
        if (c - 1, r) not in cells:
            segs.add(norm(((c, r), (c, r + 1))))
        if (c + 1, r) not in cells:
            segs.add(norm(((c + 1, r), (c + 1, r + 1))))
    return segs


def read_layout(path=LAYOUT):
    """The layout file, as floor squares and the panels the walls are made of."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if float(data.get("cell_mm", CELL_MM)) != CELL_MM:
        raise ValueError(
            f"{Path(path).name} was drawn on a {data['cell_mm']} mm grid but "
            f"this model is built for {CELL_MM:.0f} mm")
    cells = {tuple(c) for c in data["cells"]}
    added = {norm(s) for s in data.get("walls_added", [])}
    removed = {norm(s) for s in data.get("walls_removed", [])}
    walls = (boundary_walls(cells) | added) - removed
    return cells, sorted(walls), data.get("rois", [])


def solids(cells, walls):
    """Every part of the model, as boxes, without FreeCAD anywhere in sight."""
    cols = [c for seg in walls for (c, _) in seg]
    rows = [r for seg in walls for (_, r) in seg]
    c_lo, r_hi = min(cols), max(rows)
    width = (max(cols) - c_lo) * CELL_MM
    depth = (r_hi - min(rows)) * CELL_MM

    # The maze is centred on the plate. Where it actually sits is not recorded,
    # and the tunnel has to reach the edge for the transport cage to dock, so
    # check this against the rig before making anything from the drawing.
    ox = (PLATE_MM - width) / 2
    oy = (PLATE_MM - depth) / 2
    z0 = PLATE_THICK_MM

    def X(c):
        return ox + (c - c_lo) * CELL_MM

    def Y(r):
        return oy + (r_hi - r) * CELL_MM      # row 0 is at the top in the file

    out = [Box("baseplate", "Baseplate", 0.0, 0.0, 0.0,
               PLATE_MM, PLATE_MM, PLATE_THICK_MM)]

    inset = (CELL_MM - PANEL_WIDTH_MM) / 2
    for i, seg in enumerate(walls, start=1):
        (c0, r0), (c1, r1) = seg
        if r0 == r1:                                   # runs along X
            out.append(Box(f"panel{i:03d}", "Panels",
                           X(c0) + inset, Y(r0) - PANEL_THICK_MM / 2, z0,
                           PANEL_WIDTH_MM, PANEL_THICK_MM, PANEL_HEIGHT_MM))
        else:                                          # runs along Y
            out.append(Box(f"panel{i:03d}", "Panels",
                           X(c0) - PANEL_THICK_MM / 2, Y(r1) + inset, z0,
                           PANEL_THICK_MM, PANEL_WIDTH_MM, PANEL_HEIGHT_MM))

    posts = sorted({node for seg in walls for node in seg})
    for i, (c, r) in enumerate(posts, start=1):
        out.append(Box(f"post{i:03d}", "Posts",
                       X(c) - POST_MM / 2, Y(r) - POST_MM / 2, z0,
                       POST_MM, POST_MM, POST_LENGTH_MM))

    return out, width, depth


def parts_list(boxes):
    """How many of each part, for ordering and for checking against the shelf."""
    counts = {}
    for b in boxes:
        counts[b.group] = counts.get(b.group, 0) + 1
    return counts


# -- FreeCAD ----------------------------------------------------------------
def build(out_stem="eight_arm_maze", cut_holes=CUT_HOLES):
    """Make the document, and write it out beside this file."""
    import FreeCAD as App
    import Part

    cells, walls, _ = read_layout()
    boxes, width, depth = solids(cells, walls)

    doc = App.newDocument(out_stem)
    groups = {}
    shapes = []
    for b in boxes:
        shape = Part.makeBox(b.dx, b.dy, b.dz, App.Vector(b.x, b.y, b.z))
        if b.group == "Baseplate" and cut_holes:
            shape = shape.cut(_hole_grid(Part, App))
        obj = doc.addObject("Part::Feature", b.name)
        obj.Shape = shape
        if b.group not in groups:
            groups[b.group] = doc.addObject("App::DocumentObjectGroup",
                                            b.group)
        groups[b.group].addObject(obj)
        shapes.append(shape)

    doc.recompute()
    doc.saveAs(str(HERE / f"{out_stem}.FCStd"))

    compound = Part.makeCompound(shapes)
    compound.exportStep(str(HERE / f"{out_stem}.step"))
    compound.exportStl(str(HERE / f"{out_stem}.stl"))

    print(f"maze {width:.0f} x {depth:.0f} mm on a {PLATE_MM:.0f} mm plate")
    for group, n in sorted(parts_list(boxes).items()):
        print(f"  {n:4d}  {group}")
    print(f"wrote {out_stem}.FCStd, .step and .stl in {HERE}")
    return doc


def _hole_grid(Part, App):
    """The baseplate mounting holes, as one compound to cut in a single go."""
    holes = []
    n = int(PLATE_MM / HOLE_PITCH_MM) + 1
    for i in range(n):
        for j in range(n):
            holes.append(Part.makeCylinder(
                HOLE_DIA_MM / 2, PLATE_THICK_MM + 2,
                App.Vector(i * HOLE_PITCH_MM, j * HOLE_PITCH_MM, -1)))
    return Part.makeCompound(holes)


def describe():
    """What would be built, for checking a dimension without opening FreeCAD."""
    cells, walls, rois = read_layout()
    boxes, width, depth = solids(cells, walls)
    print(f"layout    {LAYOUT.name}: {len(cells)} floor squares, "
          f"{len(walls)} panels, {len(rois)} detection regions")
    print(f"maze      {width:.0f} x {depth:.0f} mm, "
          f"centred on a {PLATE_MM:.0f} x {PLATE_MM:.0f} mm plate")
    print(f"panels    {PANEL_HEIGHT_MM:.0f} x {PANEL_WIDTH_MM:.0f} x "
          f"{PANEL_THICK_MM:.0f} mm acrylic (height x width x thickness)")
    print(f"posts     {POST_MM:.0f} x {POST_MM:.0f} mm MakerBeam XL, "
          f"{POST_LENGTH_MM:.0f} mm long")
    print("parts")
    for group, n in sorted(parts_list(boxes).items()):
        print(f"  {n:4d}  {group}")
    on_plate = [b for b in boxes if b.group != "Baseplate"]
    print(f"extent    x {min(b.x for b in on_plate):.1f} to "
          f"{max(b.x + b.dx for b in on_plate):.1f} mm, "
          f"y {min(b.y for b in on_plate):.1f} to "
          f"{max(b.y + b.dy for b in on_plate):.1f} mm, "
          f"z {min(b.z for b in on_plate):.1f} to "
          f"{max(b.z + b.dz for b in on_plate):.1f} mm")


def invoked_directly():
    """Was this file asked for, or has something else merely imported it?

    The usual test, __name__ == "__main__", is not enough here. FreeCADCmd
    imports a script rather than running it, so __name__ is the module name and
    the usual test never fires. Asking whether this file is named on the
    command line answers the question for FreeCADCmd, for plain python, and for
    the test suite, which imports it and must not set a build going.
    """
    import sys
    here = Path(__file__).resolve()
    for arg in sys.argv:
        try:
            if Path(arg).resolve() == here:
                return True
        except (OSError, ValueError):
            continue
    return __name__ == "__main__"


if invoked_directly():
    try:
        import FreeCAD  # noqa: F401
    except ImportError:
        describe()
        print()
        print("FreeCAD is not on this interpreter, so nothing was built.")
        print("Run this file inside FreeCAD, or with FreeCADCmd, to get the")
        print("FCStd, STEP and STL.")
    else:
        build()
