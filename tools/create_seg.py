#!/usr/bin/env python3
"""Create and visualize 8-bit encoded semantic segmentation masks.

Converts X-AnyLabeling JSON polygon labels to single-channel uint8 PNG masks.
Standalone script — requires only numpy and opencv-python.

Modes:
    --convert   Scan all scenarios under root, convert JSON labels to PNG masks,
                and collect them into a single destination folder.
    --viz       Visualize converted masks overlaid on original images.
                Requires opencv-python (not headless) for cv2.imshow.

Output naming: <scenario>_<camera>_<timestamp>.png
    e.g. a1_scenario_00_1_0.png

Mask encoding: single-channel uint8 PNG, background=0.
Load example:
    seg = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # (H, W), uint8

Expected directory structure:
    <root>/
        <scenario_00>/
            <subdir>/          # default: "Noon"
                CAMERA_1/
                    0.jpg
                    0.json     # X-AnyLabeling label
                    100.jpg
                    ...
                CAMERA_2/
                ...
        <scenario_01>/
        ...

Dependencies:
    pip install numpy opencv-python
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Class mapping ──────────────────────────────────────────
# label name -> pixel value (trainId). Background = 0.
# Modify this dict to match your label set.
CLASS_MAP = {
    "drivable_region": 1,
    "car": 2,
    "ego": 3,
    "fence": 4,
}

# ── Visualization palette ─────────────────────────────────
# trainId -> (B, G, R), label name
PALETTE = {
    0: ((0, 0, 0),          "background"),
    1: ((200, 50, 150),     "drivable_region"),
    2: ((0, 0, 255),        "car"),
    3: ((255, 0, 0),        "ego"),
    4: ((0, 200, 0),        "fence"),
}
DEFAULT_COLOR = (255, 255, 255)


# ── Convert ────────────────────────────────────────────────

def json_to_mask(json_path: Path) -> np.ndarray | None:
    """Convert JSON label to mask. Returns None if unknown classes are found."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = {s["label"] for s in data["shapes"] if s["shape_type"] == "polygon"}
    if not labels:
        print(f"  [WARNING] {json_path.name}: no polygon labels -- skipped")
        return None
    unknown = labels - CLASS_MAP.keys()
    if unknown:
        print(f"  [WARNING] {json_path.name}: unknown classes {unknown} -- skipped")
        return None

    h, w = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    shapes = sorted(
        data["shapes"],
        key=lambda s: cv2.contourArea(np.array(s["points"], dtype=np.float32)),
        reverse=True,
    )
    for shape in shapes:
        if shape["shape_type"] != "polygon":
            continue
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], CLASS_MAP[shape["label"]])

    return mask


def find_camera_dirs(root: Path, subdir: str, cameras: list[int]) -> list[tuple[str, int, Path]]:
    """Return list of (scenario_name, camera_number, camera_dir_path)."""
    results = []
    for scenario_dir in sorted(root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        for cam_num in cameras:
            cam_dir = scenario_dir / subdir / f"CAMERA_{cam_num}"
            if cam_dir.is_dir():
                results.append((scenario_dir.name, cam_num, cam_dir))
    return results


def convert(root: Path, dst: Path, subdir: str, cameras: list[int]):
    dst.mkdir(parents=True, exist_ok=True)
    cam_entries = find_camera_dirs(root, subdir, cameras)

    if not cam_entries:
        print(f"No CAMERA directories found under {root}/*/{subdir}")
        sys.exit(1)

    total, converted = 0, 0
    for scenario, cam_num, cam_dir in cam_entries:
        json_files = sorted(cam_dir.glob("*.json"))
        if not json_files:
            continue
        print(f"[{scenario}/CAMERA_{cam_num}] {len(json_files)} label(s)")
        for jf in json_files:
            total += 1
            timestamp = jf.stem
            out_name = f"{scenario}_{cam_num}_{timestamp}.png"
            out_path = dst / out_name

            mask = json_to_mask(jf)
            if mask is None:
                continue
            cv2.imwrite(str(out_path), mask)
            converted += 1
            print(f"  {jf.name} -> {out_name}  classes: {np.unique(mask).tolist()}")

    print(f"\nDone: {converted}/{total} converted -> {dst}")


# ── Visualize ──────────────────────────────────────────────

def colorize(seg: np.ndarray) -> np.ndarray:
    h, w = seg.shape[:2]
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for tid in np.unique(seg):
        bgr = PALETTE[tid][0] if tid in PALETTE else DEFAULT_COLOR
        color[seg == tid] = bgr
    return color


def build_legend(seg: np.ndarray, scale: int = 20) -> np.ndarray:
    present = sorted(np.unique(seg))
    entries = []
    for tid in present:
        name = PALETTE[tid][1] if tid in PALETTE else f"class_{tid}"
        bgr = PALETTE[tid][0] if tid in PALETTE else DEFAULT_COLOR
        entries.append((tid, name, bgr))

    legend_h = len(entries) * (scale + 5) + 10
    legend_w = 200
    legend = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
    for i, (tid, name, bgr) in enumerate(entries):
        y = 10 + i * (scale + 5)
        cv2.rectangle(legend, (5, y), (5 + scale, y + scale), bgr, -1)
        cv2.putText(legend, f"{tid}: {name}", (scale + 12, y + scale - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return legend


def parse_mask_filename(name: str) -> tuple[str, int, str] | None:
    """Parse <scenario>_<camera>_<timestamp>.png -> (scenario, cam_num, timestamp)."""
    parts = name.removesuffix(".png").rsplit("_", 2)
    if len(parts) != 3:
        return None
    scenario, cam_str, timestamp = parts
    if not cam_str.isdigit():
        return None
    return scenario, int(cam_str), timestamp


def render_frame(root: Path, subdir: str, mf: Path) -> np.ndarray | None:
    """Render a single overlay frame. Returns canvas or None on failure."""
    parsed = parse_mask_filename(mf.name)
    if parsed is None:
        return None

    scenario, cam_num, timestamp = parsed
    img_path = root / scenario / subdir / f"CAMERA_{cam_num}" / f"{timestamp}.jpg"
    if not img_path.exists():
        return None

    img = cv2.imread(str(img_path))
    seg = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
    if img is None or seg is None:
        return None

    color_mask = colorize(seg)
    if color_mask.shape[:2] != img.shape[:2]:
        color_mask = cv2.resize(color_mask, (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        seg = cv2.resize(seg, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    # Background: heavy color overlay (0.9), labeled regions: lighter overlay (0.5)
    bg_mask = (seg == 0)
    overlay = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)
    overlay[bg_mask] = cv2.addWeighted(img, 0.1, color_mask, 0.9, 0)[bg_mask]
    legend = build_legend(seg)

    if legend.shape[0] < overlay.shape[0]:
        pad = np.zeros((overlay.shape[0] - legend.shape[0], legend.shape[1], 3), dtype=np.uint8)
        legend = np.vstack([legend, pad])
    else:
        legend = legend[:overlay.shape[0]]

    return np.hstack([overlay, legend])


def viz(root: Path, dst: Path, subdir: str):
    mask_files = sorted(dst.glob("*.png"))
    if not mask_files:
        print(f"No mask files found in {dst}")
        sys.exit(1)

    valid = [(i, mf) for i, mf in enumerate(mask_files) if parse_mask_filename(mf.name)]
    if not valid:
        print("No valid mask files found.")
        sys.exit(1)

    print(f"Found {len(valid)} mask(s) in {dst}")
    print("  Arrow Right = next, Arrow Left = prev, ESC = quit")

    KEY_ESC = 27
    KEY_RIGHT_GTK = 83
    KEY_RIGHT_QT = 65363
    KEY_LEFT_GTK = 81
    KEY_LEFT_QT = 65361

    idx = 0
    win_name = "create_seg - viz"
    prev_idx = -1
    while True:
        idx = max(0, min(idx, len(valid) - 1))
        if idx != prev_idx:
            _, mf = valid[idx]
            canvas = render_frame(root, subdir, mf)
            if canvas is not None:
                cv2.imshow(win_name, canvas)
                cv2.setWindowTitle(win_name, f"[{idx+1}/{len(valid)}] {mf.name}")
                print(f"  [{idx+1}/{len(valid)}] {mf.name}")
            prev_idx = idx

        key = cv2.waitKeyEx(0) & 0xFFFF
        if key == KEY_ESC:
            break
        elif key in (KEY_LEFT_GTK, KEY_LEFT_QT):
            idx -= 1
        elif key in (KEY_RIGHT_GTK, KEY_RIGHT_QT):
            idx += 1

    cv2.destroyAllWindows()


# ── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--convert", action="store_true", help="convert JSON labels to PNG masks")
    mode.add_argument("--viz", action="store_true", help="visualize converted masks")

    parser.add_argument("--root", type=str, required=True,
                        help="root directory containing scenario folders")
    parser.add_argument("--dst", type=str, default=None,
                        help="destination folder for masks (default: <root>/seg_labels)")
    parser.add_argument("--subdir", type=str, default="Noon",
                        help="sub-directory under each scenario (default: Noon)")
    parser.add_argument("--cameras", type=str, default="1,2,3,4,5",
                        help="comma-separated camera numbers (default: 1,2,3,4,5)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    dst = Path(args.dst) if args.dst else root / "seg_labels"
    cameras = [int(c) for c in args.cameras.split(",")]

    if args.convert:
        convert(root, dst, args.subdir, cameras)
    elif args.viz:
        viz(root, dst, args.subdir)


if __name__ == "__main__":
    main()
