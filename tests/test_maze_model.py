"""The eight-arm maze solid model.

The model in hardware/3dmodels/eight_arm_maze is generated from a layout file
rather than drawn, so the arithmetic that turns squares on a grid into panels
and posts is code, and can be wrong. The half of it that needs FreeCAD cannot
run here; the half that works out the geometry can, and that is the half that
holds the dimensions.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODEL_DIR = (Path(__file__).resolve().parents[1]
             / "hardware" / "3dmodels" / "eight_arm_maze")


@pytest.fixture(scope="module")
def model():
    """The generator, imported by path: it is a FreeCAD macro, not a module."""
    spec = importlib.util.spec_from_file_location(
        "build_maze_model", MODEL_DIR / "build_maze_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write(tmp_path, name, **layout):
    layout.setdefault("cell_mm", 50)
    path = tmp_path / name
    path.write_text(json.dumps(layout), encoding="utf-8")
    return path


def test_one_square_gives_four_panels_and_four_posts(model, tmp_path):
    path = write(tmp_path, "one.json", cells=[[2, 2]])
    cells, walls, _ = model.read_layout(path)
    assert len(walls) == 4

    boxes, width, depth = model.solids(cells, walls)
    assert (width, depth) == (50.0, 50.0)
    assert model.parts_list(boxes) == {"Baseplate": 1, "Panels": 4, "Posts": 4}

    corners = sorted((b.x + b.dx / 2, b.y + b.dy / 2)
                     for b in boxes if b.group == "Posts")
    assert corners == [(325.0, 325.0), (325.0, 375.0),
                       (375.0, 325.0), (375.0, 375.0)]


def test_an_opening_removes_its_panel(model, tmp_path):
    path = write(tmp_path, "open.json", cells=[[2, 2]],
                 walls_removed=[[[2, 3], [3, 3]]])
    _, walls, _ = model.read_layout(path)
    assert len(walls) == 3


def test_a_divider_between_two_squares_is_kept(model, tmp_path):
    """Floor on both sides, so the boundary rule cannot find it."""
    divider = [[3, 2], [3, 3]]
    path = write(tmp_path, "pair.json", cells=[[2, 2], [3, 2]],
                 walls_added=[divider])
    _, walls, _ = model.read_layout(path)
    assert len(walls) == 7                       # six outside, one between
    assert model.norm(divider) in walls


def test_a_layout_on_another_grid_is_refused(model, tmp_path):
    path = tmp_path / "odd.json"
    path.write_text(json.dumps({"cell_mm": 40, "cells": [[0, 0]]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="40"):
        model.read_layout(path)


def test_parts_stand_on_the_plate(model):
    cells, walls, _ = model.read_layout()
    boxes, _, _ = model.solids(cells, walls)
    for box in boxes:
        if box.group == "Baseplate":
            continue
        assert box.z == model.PLATE_THICK_MM
        assert box.dz in (model.PANEL_HEIGHT_MM, model.POST_LENGTH_MM)


def test_panels_span_one_post_pitch(model):
    """A panel is as wide as the post spacing, because it seats in the slots.

    It therefore overlaps each post it meets by half a post. That is the panel
    sitting in the T-slot, not a clash.
    """
    cells, walls, _ = model.read_layout()
    boxes, _, _ = model.solids(cells, walls)
    panels = [b for b in boxes if b.group == "Panels"]
    assert panels
    for box in panels:
        assert max(box.dx, box.dy) == model.PANEL_WIDTH_MM
        assert min(box.dx, box.dy) == model.PANEL_THICK_MM
    assert model.PANEL_WIDTH_MM == model.CELL_MM


def test_the_posts_are_tall_enough_for_the_panels(model):
    """A 100 mm panel needs a 100 mm post to hold it."""
    assert model.POST_LENGTH_MM >= model.PANEL_HEIGHT_MM


def test_the_maze_lands_on_the_plate_with_nothing_duplicated(model):
    cells, walls, _ = model.read_layout()
    boxes, width, depth = model.solids(cells, walls)
    assert width <= model.PLATE_MM and depth <= model.PLATE_MM

    for box in boxes:
        assert 0 <= box.x and box.x + box.dx <= model.PLATE_MM
        assert 0 <= box.y and box.y + box.dy <= model.PLATE_MM

    for group in ("Posts", "Panels"):
        placed = [(b.x, b.y, b.z) for b in boxes if b.group == group]
        assert len(placed) == len(set(placed))


def test_the_layout_shipped_with_the_model_is_the_eight_arm_maze(model):
    """If the layout is redrawn, this is the test that should be updated."""
    cells, walls, rois = model.read_layout()
    boxes, width, depth = model.solids(cells, walls)
    assert (width, depth) == (350.0, 300.0)
    assert len(rois) == 8
    assert model.parts_list(boxes) == {"Baseplate": 1, "Panels": 37,
                                       "Posts": 38}
