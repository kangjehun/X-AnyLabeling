from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from anylabeling.services.auto_labeling.depth_anything_v2 import (
    DepthAnythingV2,
    depth_output_path,
)


class DepthAnythingV2OutputTest(unittest.TestCase):
    def test_dataset_depth_path_preserves_relative_camera_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "RUN10"
            image_path = run_dir / "images" / "front" / "nested" / "123.jpg"
            expected = (
                run_dir / "depth" / "front" / "nested" / "123_depth.npy"
            )
            self.assertEqual(depth_output_path(image_path), expected)

    def test_non_dataset_image_uses_local_depth_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "CAMERA_1" / "123.jpg"
            expected = image_path.parent / "depth" / "123_depth.npy"
            self.assertEqual(depth_output_path(image_path), expected)

    def test_postprocess_returns_finite_float32_for_constant_prediction(self):
        model = DepthAnythingV2.__new__(DepthAnythingV2)
        model.render_mode = "gray"
        model.min_depth = 0.0
        model.max_depth = 1.0
        prediction = np.ones((1, 2, 3), dtype=np.float32)

        _, depth = model.postprocess(prediction, (4, 6))

        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(depth.shape, (4, 6))
        self.assertTrue(np.all(np.isfinite(depth)))
        self.assertTrue(np.all(depth == 0.0))

    def test_raw_depth_mode_saves_only_npy_in_dataset_depth_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "RUN10"
            image_path = run_dir / "images" / "front" / "123.jpg"
            model = DepthAnythingV2.__new__(DepthAnythingV2)
            model.file_ext = ".png"
            model.save_raw_depth = True
            model.preprocess = lambda image: (None, (4, 6))
            model.forward = lambda blob: None
            expected_depth = np.full((4, 6), 0.5, dtype=np.float32)
            model.postprocess = lambda outputs, shape: (
                np.zeros((4, 6), dtype=np.uint8),
                expected_depth,
            )

            with patch(
                "anylabeling.services.auto_labeling.depth_anything_v2."
                "qt_img_to_rgb_cv_img",
                return_value=np.zeros((4, 6, 3), dtype=np.uint8),
            ):
                model.predict_shapes(object(), str(image_path))

            output_path = run_dir / "depth" / "front" / "123_depth.npy"
            self.assertTrue(output_path.is_file())
            self.assertFalse(output_path.with_name("123.png").exists())
            np.testing.assert_array_equal(np.load(output_path), expected_depth)


if __name__ == "__main__":
    unittest.main()
