"""
slp_export.py

Turn a sleap-nn `*.predictions.slp` file into the two derived artifacts most
analyses actually want:

  1. A per-frame keypoints CSV  -> `<stem>.keypoints.csv`
     One row per video frame (0 .. N-1). Columns:
       frame_idx, instance_score,
       <node>.x, <node>.y, <node>.score   (for every skeleton node)
     Frames where no mouse was detected are written as empty/NaN, so the row
     index lines up 1:1 with the source video timeline.

  2. (optional) An annotated video -> `<stem>.labeled.mp4`
     The source frames with keypoints + skeleton edges drawn on top. Rendering
     is OFF by default because a full ~100k-frame clip is slow and large; use
     `--render` and (recommended) `--render_frames N` to make a short preview.

Run with the sleap-nn interpreter (it has sleap_io + cv2), e.g.:

  & "$env:APPDATA\\uv\\tools\\sleap-nn\\Scripts\\python.exe" ^
      analysis\\slp_export.py ^
      --slp "D:\\simplermaze_output\\...\\10231_..._cropped.predictions.slp" ^
      --render --render_frames 600

Batch all predictions under a folder (CSV only is the cheap default):

  & "...python.exe" analysis\\slp_export.py --input_dir D:\\simplermaze_output
"""

import argparse
import csv
import sys
from pathlib import Path


# 10-colour BGR palette (cv2 uses BGR), reused cyclically for nodes.
_PALETTE_BGR = [
    (66, 99, 235), (80, 175, 76), (227, 119, 31), (40, 39, 214),
    (189, 103, 148), (75, 86, 140), (194, 119, 227), (127, 127, 127),
    (34, 189, 188), (207, 190, 23),
]


def load_labels(slp_path):
    import sleap_io as sio
    return sio.load_slp(str(slp_path))


def frame_index_map(labels):
    """Map frame_idx -> first instance (single-animal) for quick lookup."""
    by_idx = {}
    for lf in labels.labeled_frames:
        if len(lf.instances):
            by_idx[lf.frame_idx] = lf.instances[0]
    return by_idx


def video_frame_count(labels, video_path):
    """Total frames in the source video (falls back to max labeled idx + 1)."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    except Exception:
        pass
    idxs = [lf.frame_idx for lf in labels.labeled_frames]
    return (max(idxs) + 1) if idxs else 0


def export_csv(labels, csv_path):
    """Write the per-frame wide CSV. Returns (n_rows, n_with_instance)."""
    nodes = [n.name for n in labels.skeleton.nodes]
    by_idx = frame_index_map(labels)
    video_path = labels.videos[0].filename if labels.videos else None
    total = video_frame_count(labels, video_path)

    header = ["frame_idx", "instance_score"]
    for nd in nodes:
        header += [f"{nd}.x", f"{nd}.y", f"{nd}.score"]

    n_with = 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for fi in range(total):
            inst = by_idx.get(fi)
            if inst is None:
                w.writerow([fi, ""] + [""] * (3 * len(nodes)))
                continue
            n_with += 1
            xyz = inst.numpy(scores=True)  # (n_nodes, 3): x, y, score
            score = getattr(inst, "score", "")
            row = [fi, "" if score is None else round(float(score), 5)]
            for x, y, s in xyz:
                row += [
                    "" if x != x else round(float(x), 3),   # x!=x -> NaN
                    "" if y != y else round(float(y), 3),
                    "" if s != s else round(float(s), 5),
                ]
            w.writerow(row)
    return total, n_with


def render_video(labels, video_path, out_path, start, n_frames,
                 radius, thickness, score_thresh):
    """Draw keypoints + skeleton edges onto frames -> annotated mp4."""
    import cv2

    nodes = [n.name for n in labels.skeleton.nodes]
    name_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(name_to_idx[e.source.name], name_to_idx[e.destination.name])
             for e in labels.skeleton.edges]
    by_idx = frame_index_map(labels)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open source video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    end = total if n_frames is None else min(total, start + n_frames)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    fi = start
    written = 0
    while fi < end:
        ok, frame = cap.read()
        if not ok:
            break
        inst = by_idx.get(fi)
        if inst is not None:
            xy = inst.numpy(scores=True)  # (n, 3)
            pts = {}
            for ni, (x, y, s) in enumerate(xy):
                if x == x and y == y and s >= score_thresh:  # x==x -> not NaN
                    pts[ni] = (int(round(x)), int(round(y)))
            for a, b in edges:
                if a in pts and b in pts:
                    cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1,
                             cv2.LINE_AA)
            for ni, p in pts.items():
                cv2.circle(frame, p, radius,
                           _PALETTE_BGR[ni % len(_PALETTE_BGR)], thickness,
                           cv2.LINE_AA)
        cv2.putText(frame, f"frame {fi}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        writer.write(frame)
        fi += 1
        written += 1

    cap.release()
    writer.release()
    return written, fps


def find_slps(input_dir):
    return sorted(Path(input_dir).rglob("*.predictions.slp"))


def process_one(slp_path, args):
    slp_path = Path(slp_path)
    # `<name>.predictions.slp` -> base `<name>`
    base = slp_path.name[:-len(".predictions.slp")] \
        if slp_path.name.endswith(".predictions.slp") else slp_path.stem

    labels = load_labels(slp_path)
    video_path = Path(args.video) if args.video else (
        Path(labels.videos[0].filename) if labels.videos else None)

    if not args.no_csv:
        csv_path = Path(args.csv) if args.csv else slp_path.parent / f"{base}.keypoints.csv"
        total, n_with = export_csv(labels, csv_path)
        print(f"  CSV : {csv_path}  ({total} frames, {n_with} with a mouse)")

    if args.render:
        if video_path is None or not Path(video_path).exists():
            print(f"  SKIP render: source video not found ({video_path})",
                  file=sys.stderr)
        else:
            out_path = Path(args.video_out) if args.video_out else \
                slp_path.parent / f"{base}.labeled.mp4"
            written, fps = render_video(
                labels, video_path, out_path,
                start=args.render_start, n_frames=args.render_frames,
                radius=args.point_radius, thickness=args.thickness,
                score_thresh=args.score_thresh)
            print(f"  VID : {out_path}  ({written} frames @ {fps:.1f} fps)")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--slp", type=Path, help="A single *.predictions.slp file.")
    src.add_argument("--input_dir", type=Path,
                     help="Folder to search recursively for *.predictions.slp.")
    p.add_argument("--video", type=Path, default=None,
                   help="Override source video path (default: read from the .slp).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Output CSV path (single-file mode only).")
    p.add_argument("--video_out", type=Path, default=None,
                   help="Output annotated mp4 path (single-file mode only).")
    p.add_argument("--no_csv", action="store_true", help="Skip CSV export.")
    p.add_argument("--render", action="store_true",
                   help="Also render an annotated mp4 (off by default).")
    p.add_argument("--render_start", type=int, default=0,
                   help="First frame to render (default 0).")
    p.add_argument("--render_frames", type=int, default=None,
                   help="How many frames to render (default: all — large!). "
                        "Use e.g. 600 for a ~quick preview clip.")
    p.add_argument("--score_thresh", type=float, default=0.0,
                   help="Don't draw points below this confidence (default 0).")
    p.add_argument("--point_radius", type=int, default=3)
    p.add_argument("--thickness", type=int, default=-1,
                   help="Circle thickness; -1 = filled (default).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.slp:
        targets = [args.slp]
    else:
        targets = find_slps(args.input_dir)
        if (args.csv or args.video_out):
            print("Error: --csv/--video_out only apply with --slp (single file).",
                  file=sys.stderr)
            return 1
        print(f"Found {len(targets)} prediction file(s) under {args.input_dir}.")
    for i, slp in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {Path(slp).name}")
        try:
            process_one(slp, args)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
