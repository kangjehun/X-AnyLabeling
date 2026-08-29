import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.generate_depth_dataset import discover_samples, validate_depth_file


class GenerateDepthDatasetTest(unittest.TestCase):
    def test_discovery_preserves_camera_paths_and_ignores_delete_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "RUN10"
            image_dir = run / "images" / "front"
            json_dir = run / "labels_json" / "front"
            image_dir.mkdir(parents=True)
            json_dir.mkdir(parents=True)
            image = np.zeros((4, 6, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_dir / "123.jpg"), image))
            (json_dir / "123.json").write_text(
                json.dumps(
                    {
                        "imagePath": "123.jpg",
                        "imageHeight": 4,
                        "imageWidth": 6,
                        "shapes": [],
                    }
                ),
                encoding="utf-8",
            )
            deleted = run / "images" / "_delete_"
            deleted.mkdir()
            self.assertTrue(cv2.imwrite(str(deleted / "old.jpg"), image))

            samples = discover_samples(root, groups=("runs",))

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].relative_stem, Path("front/123"))
            self.assertEqual(
                samples[0].output_path,
                run / "depth" / "front" / "123_depth.npy",
            )

    def test_depth_validation_checks_dtype_shape_finiteness_and_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "depth.npy"
            np.save(path, np.full((4, 6), 0.5, dtype=np.float32))
            valid, _ = validate_depth_file(path, (4, 6))
            self.assertTrue(valid)

            np.save(path, np.full((4, 6), 2.0, dtype=np.float32))
            valid, detail = validate_depth_file(path, (4, 6))
            self.assertFalse(valid)
            self.assertIn("expected [0, 1]", detail)


if __name__ == "__main__":
    unittest.main()
