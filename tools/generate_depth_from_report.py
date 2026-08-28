#!/usr/bin/env python3
"""Generate relative depth labels from a segmentation-only report.

Parses the item table in `segmentation_only_items.txt`, resolves the paired
image for each JSON label, and runs the repository's patched
`Depth Anything V2 (ViT-Large)` model directly without opening the GUI.

Output path follows the current local X-AnyLabeling modification:
    <run>/images/<camera>/{timestamp}.jpg
    -> <run>/depth/<camera>/{timestamp}_depth.npy

Example:
    python tools/generate_depth_from_report.py \
      --report "/media/.../label_check_reports/segmentation_only_items.txt"
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtGui import QImage


# Match app bootstrap to avoid threading issues in some BLAS backends.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anylabeling.services.auto_labeling.depth_anything_v2 import (  # noqa: E402
    DepthAnythingV2,
    depth_output_path,
)
from anylabeling.config import set_work_directory  # noqa: E402
from anylabeling import config as anylabeling_config  # noqa: E402


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class ReportItem:
    scenario: str
    camera: str
    timestamp: str
    json_path: Path


def parse_report(report_path: Path) -> list[ReportItem]:
    items: list[ReportItem] = []
    in_table = False

    with report_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if "Scenario" in line and "Camera" in line and "JSON Path" in line:
                in_table = True
                continue
            if not in_table or "|" not in line or ".json" not in line:
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 4:
                continue

            scenario, camera, timestamp, json_path = parts[:4]
            if not scenario or not camera or not timestamp or not json_path:
                continue
            if not json_path.endswith(".json"):
                continue

            items.append(
                ReportItem(
                    scenario=scenario,
                    camera=camera,
                    timestamp=timestamp,
                    json_path=Path(json_path),
                )
            )

    if not items:
        raise ValueError(f"No report items found in {report_path}")

    return items


def resolve_image_path(json_path: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def validate_depth(npy_path: Path) -> tuple[bool, str]:
    try:
        depth = np.load(npy_path)
    except Exception as exc:
        return False, f"failed to load output: {exc}"

    if depth.dtype != np.float32:
        return False, f"dtype={depth.dtype}, expected float32"
    if depth.ndim != 2:
        return False, f"ndim={depth.ndim}, expected 2"

    dmin = float(depth.min())
    dmax = float(depth.max())
    if dmin < 0.0 or dmax > 1.0:
        return False, f"range=[{dmin:.6f}, {dmax:.6f}], expected [0, 1]"

    return True, f"shape={depth.shape}, range=[{dmin:.4f}, {dmax:.4f}]"


def load_model(config_path: Path) -> DepthAnythingV2:
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["config_file"] = str(config_path)
    return DepthAnythingV2(config, on_message=lambda msg: print(f"[model] {msg}"))


def run(
    report_path: Path,
    config_path: Path,
    work_dir: Path,
    overwrite: bool,
    dry_run: bool,
    limit: int | None,
    scenario_filter: set[str] | None,
    camera_filter: set[str] | None,
) -> int:
    items = parse_report(report_path)

    if scenario_filter:
        items = [item for item in items if item.scenario in scenario_filter]
    if camera_filter:
        items = [item for item in items if item.camera in camera_filter]
    if limit is not None:
        items = items[:limit]

    if not items:
        print("No matching items after filters.")
        return 1

    print(f"Report: {report_path}")
    print(f"Targets: {len(items)}")
    print(f"Model config: {config_path}")
    print(f"Work dir: {work_dir}")
    print(f"Mode: {'dry-run' if dry_run else 'generate'}")

    set_work_directory(str(work_dir))
    anylabeling_config.current_config_file = str(
        work_dir / ".xanylabelingrc"
    )

    model = None if dry_run else load_model(config_path)

    generated = 0
    skipped_existing = 0
    missing_images = 0
    missing_json = 0
    failed = 0

    try:
        for index, item in enumerate(items, start=1):
            prefix = f"[{index}/{len(items)}] {item.scenario} {item.camera} {item.timestamp}"

            if not item.json_path.exists():
                print(f"{prefix} -> missing json: {item.json_path}")
                missing_json += 1
                continue

            image_path = resolve_image_path(item.json_path)
            if image_path is None:
                print(f"{prefix} -> missing image next to json")
                missing_images += 1
                continue

            output_path = depth_output_path(image_path)
            if output_path.exists() and not overwrite:
                print(f"{prefix} -> skip existing: {output_path}")
                skipped_existing += 1
                continue

            if dry_run:
                print(f"{prefix} -> would generate: {output_path}")
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                model.predict_shapes(QImage(), str(image_path))
            except Exception as exc:
                print(f"{prefix} -> inference failed: {exc}")
                failed += 1
                continue

            if not output_path.exists():
                print(f"{prefix} -> output not created: {output_path}")
                failed += 1
                continue

            is_valid, detail = validate_depth(output_path)
            if not is_valid:
                print(f"{prefix} -> invalid output: {detail}")
                failed += 1
                continue

            print(f"{prefix} -> ok: {detail}")
            generated += 1
    finally:
        if model is not None:
            model.unload()

    print("\nSummary")
    print(f"  generated:       {generated}")
    print(f"  skipped_existing:{skipped_existing}")
    print(f"  missing_json:    {missing_json}")
    print(f"  missing_images:  {missing_images}")
    print(f"  failed:          {failed}")

    return 0 if failed == 0 and missing_images == 0 and missing_json == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate relative depth labels from a segmentation-only report."
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to segmentation_only_items.txt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "anylabeling/configs/auto_labeling/depth_anything_v2_vit_l.yaml",
        help="Depth model YAML config path",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.home(),
        help="X-AnyLabeling work directory used for .xanylabelingrc and model cache",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even if *_depth.npy already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and resolve targets without running inference",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N matched items",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Filter by scenario name, repeatable",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Filter by camera name such as CAMERA_1, repeatable",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    return run(
        report_path=args.report,
        config_path=args.config,
        work_dir=args.work_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        limit=args.limit,
        scenario_filter=set(args.scenario) if args.scenario else None,
        camera_filter=set(args.camera) if args.camera else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
