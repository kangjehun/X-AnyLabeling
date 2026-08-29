#!/usr/bin/env python3
"""Generate validated relative-depth NPY files for structured IAC runs."""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys


os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
from PyQt6.QtGui import QImage


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anylabeling import config as anylabeling_config  # noqa: E402
from anylabeling.config import set_work_directory  # noqa: E402
from anylabeling.services.auto_labeling.depth_anything_v2 import (  # noqa: E402
    DepthAnythingV2,
    depth_output_path,
)


DEFAULT_GROUPS = ("legacy", "new_runs", "runs")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IGNORED_DIRECTORY_NAMES = {"_delete_"}


@dataclass(frozen=True)
class DepthSample:
    group: str
    run_name: str
    relative_stem: Path
    image_path: Path
    json_path: Path
    output_path: Path
    expected_shape: tuple[int, int]

    @property
    def display_name(self):
        return f"{self.group}/{self.run_name}/{self.relative_stem.as_posix()}"


def _included(path, root):
    directory_parts = path.relative_to(root).parts[:-1]
    return IGNORED_DIRECTORY_NAMES.isdisjoint(directory_parts)


def _unique_relative_stems(paths, root, kind):
    result = {}
    for path in paths:
        stem = path.relative_to(root).with_suffix("")
        if stem in result:
            raise ValueError(
                f"Duplicate {kind} stem {stem}: {result[stem]}, {path}"
            )
        result[stem] = path
    return result


def discover_samples(dataset_root, groups=DEFAULT_GROUPS, run_filters=None):
    dataset_root = Path(dataset_root).expanduser().resolve()
    run_filters = set(run_filters or ())
    samples = []
    for group in groups:
        group_dir = dataset_root / group
        if not group_dir.is_dir():
            raise FileNotFoundError(f"Dataset group not found: {group_dir}")
        for run_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            if run_filters and run_dir.name not in run_filters:
                continue
            images_dir = run_dir / "images"
            labels_json_dir = run_dir / "labels_json"
            if not images_dir.is_dir() or not labels_json_dir.is_dir():
                continue

            image_paths = sorted(
                path
                for path in images_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                    and _included(path, images_dir)
                )
            )
            json_paths = sorted(
                path
                for path in labels_json_dir.rglob("*.json")
                if _included(path, labels_json_dir)
            )
            images = _unique_relative_stems(image_paths, images_dir, "image")
            labels = _unique_relative_stems(json_paths, labels_json_dir, "JSON")
            missing_json = sorted(set(images) - set(labels), key=str)
            missing_images = sorted(set(labels) - set(images), key=str)
            if missing_json or missing_images:
                raise ValueError(
                    f"{group}/{run_dir.name}: missing JSON={len(missing_json)}, "
                    f"missing image={len(missing_images)}"
                )

            for relative_stem in sorted(images, key=str):
                image_path = images[relative_stem]
                json_path = labels[relative_stem]
                with json_path.open("r", encoding="utf-8") as handle:
                    annotation = json.load(handle)
                recorded_name = Path(str(annotation.get("imagePath", ""))).name
                if recorded_name != image_path.name:
                    raise ValueError(
                        f"{json_path}: imagePath basename {recorded_name!r} "
                        f"does not match {image_path.name!r}"
                    )
                expected_shape = (
                    int(annotation["imageHeight"]),
                    int(annotation["imageWidth"]),
                )
                output_path = depth_output_path(image_path)
                expected_root = (run_dir / "depth").resolve()
                if (
                    output_path.parent != expected_root
                    and expected_root not in output_path.parents
                ):
                    raise ValueError(
                        f"Depth output escaped run directory: {output_path}"
                    )
                samples.append(
                    DepthSample(
                        group=group,
                        run_name=run_dir.name,
                        relative_stem=relative_stem,
                        image_path=image_path,
                        json_path=json_path,
                        output_path=output_path,
                        expected_shape=expected_shape,
                    )
                )
    if not samples:
        raise ValueError("No matching image/JSON samples found")
    return samples


def validate_depth_file(path, expected_shape):
    try:
        depth = np.load(path, allow_pickle=False, mmap_mode="r")
    except Exception as error:
        return False, f"failed to load: {error}"
    if depth.dtype != np.float32:
        return False, f"dtype={depth.dtype}, expected float32"
    if depth.ndim != 2:
        return False, f"ndim={depth.ndim}, expected 2"
    if depth.shape != tuple(expected_shape):
        return False, f"shape={depth.shape}, expected {tuple(expected_shape)}"
    if not np.all(np.isfinite(depth)):
        return False, "contains NaN or infinity"
    minimum = float(depth.min())
    maximum = float(depth.max())
    if minimum < 0.0 or maximum > 1.0:
        return False, f"range=[{minimum:.6f}, {maximum:.6f}], expected [0, 1]"
    return True, f"shape={depth.shape}, range=[{minimum:.4f}, {maximum:.4f}]"


def load_model(config_path):
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected = {
        "type": "depth_anything_v2",
        "display_name": "Depth Anything V2 (ViT-Large)",
        "min_depth": 0.0,
        "max_depth": 1.0,
        "save_raw_depth": True,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Depth config is not the required ViT-L setup: {mismatches}")
    config["config_file"] = str(config_path)
    model = DepthAnythingV2(
        config,
        on_message=lambda message: print(f"[model] {message}", flush=True),
    )
    providers = model.net.ort_session.get_providers()
    if "CUDAExecutionProvider" not in providers:
        model.unload()
        raise RuntimeError(
            "CUDAExecutionProvider is not active; refusing a silent CPU "
            f"fallback for batch generation. Active providers: {providers}"
        )
    print(f"Active ONNX providers: {providers}", flush=True)
    return model


def generate_depths(
    samples,
    config_path,
    work_dir,
    *,
    overwrite=False,
    dry_run=False,
    progress_every=10,
):
    set_work_directory(str(work_dir))
    anylabeling_config.current_config_file = str(
        Path(work_dir) / ".xanylabelingrc"
    )
    model = None if dry_run else load_model(config_path)
    generated = 0
    skipped = 0
    failed = 0
    try:
        for index, sample in enumerate(samples, 1):
            if sample.output_path.exists() and not overwrite:
                valid, detail = validate_depth_file(
                    sample.output_path, sample.expected_shape
                )
                if valid:
                    skipped += 1
                    if index % progress_every == 0 or index == len(samples):
                        print(
                            f"[{index}/{len(samples)}] valid existing; "
                            f"generated={generated} skipped={skipped} failed={failed}",
                            flush=True,
                        )
                    continue
                print(
                    f"[INVALID EXISTING] {sample.display_name}: {detail}",
                    flush=True,
                )
                failed += 1
                continue

            if dry_run:
                generated += 1
                continue

            try:
                model.predict_shapes(QImage(), str(sample.image_path))
                valid, detail = validate_depth_file(
                    sample.output_path, sample.expected_shape
                )
                if not valid:
                    raise ValueError(detail)
                generated += 1
            except Exception as error:
                failed += 1
                print(
                    f"[FAILED] {sample.display_name}: {error}",
                    flush=True,
                )
            if index % progress_every == 0 or index == len(samples):
                print(
                    f"[{index}/{len(samples)}] generated={generated} "
                    f"skipped={skipped} failed={failed}",
                    flush=True,
                )
    finally:
        if model is not None:
            model.unload()
    return {
        "samples": len(samples),
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--group",
        action="append",
        choices=DEFAULT_GROUPS,
        help="Dataset group to process; repeatable (default: all three).",
    )
    parser.add_argument(
        "--run",
        action="append",
        help="Run directory name to process; repeatable (default: all).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "anylabeling/configs/auto_labeling/depth_anything_v2_vit_l.yaml",
    )
    parser.add_argument("--work-dir", type=Path, default=Path.home())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")
    groups = tuple(args.group) if args.group else DEFAULT_GROUPS
    samples = discover_samples(args.dataset_root, groups, args.run)
    if args.limit is not None:
        samples = samples[: args.limit]
    print(
        f"Discovered {len(samples)} samples from groups={groups}; "
        f"mode={'dry-run' if args.dry_run else 'generate'}",
        flush=True,
    )
    summary = generate_depths(
        samples,
        args.config.expanduser().resolve(),
        args.work_dir.expanduser().resolve(),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        progress_every=args.progress_every,
    )
    print(f"Summary: {summary}", flush=True)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
