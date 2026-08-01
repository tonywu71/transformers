# Copyright 2026 H Company and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the NeoMME image processor."""

import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image

    from transformers import NeoMMEImageProcessor, NeoMMEImageProcessorPil


class NeoMMEImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        do_rescale=True,
        rescale_factor=1 / 127.5,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        patch_size=4,
        max_side=None,
        max_pixels=None,
        min_pixels=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        # `image_mean` is a SHIFT and `image_std` a no-op here: together with rescale_factor they express
        # `pixel / 127.5 - 1`. They are not dataset statistics.
        self.image_mean = image_mean if image_mean is not None else [1.0, 1.0, 1.0]
        self.image_std = image_std if image_std is not None else [1.0, 1.0, 1.0]
        self.patch_size = patch_size
        self.max_side = max_side
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

    def prepare_image_processor_dict(self):
        """The canonical init kwargs.

        The resolution budgets are deliberately absent: they default to `None`, and `to_dict()` drops
        None-valued attributes library-wide, so listing them here would make the mixin's json round-trip
        test look for keys that never get written. They are exercised explicitly instead, with real values,
        by `test_no_resize_by_default_and_the_budgets_bound_the_grid`.
        """
        return {
            "do_resize": self.do_resize,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "patch_size": self.patch_size,
        }

    def expected_num_patches(self, image) -> int:
        """Patch count for one image at native resolution: the grid is ceil(side / patch_size)."""
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2] if image.shape[-1] in (1, 3, 4) else image.shape[-2:]
        else:
            height, width = image.shape[-2:]
        return -(-height // self.patch_size) * (-(-width // self.patch_size))

    def expected_output_image_shape(self, images) -> tuple[int, int]:
        """`pixel_values` is FLAT: patches of every image concatenated, never padded to a common count."""
        return sum(self.expected_num_patches(image) for image in images), 3 * self.patch_size**2

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class NeoMMEImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = NeoMMEImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            for attribute in ("do_resize", "do_rescale", "rescale_factor", "do_normalize", "patch_size"):
                self.assertTrue(hasattr(image_processing, attribute))
            for attribute in ("max_side", "max_pixels", "min_pixels"):
                self.assertTrue(hasattr(image_processing, attribute))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.patch_size, self.image_processor_tester.patch_size)
            self.assertIsNone(image_processor.max_side)  # no budget unless asked for: native resolution

            image_processor = image_processing_class.from_dict(self.image_processor_dict, patch_size=8, max_side=64)
            self.assertEqual(image_processor.patch_size, 8)
            self.assertEqual(image_processor.max_side, 64)

    # --- the three call tests, overridden because `pixel_values` is flat rather than batched ---

    def _check_call(self, image_inputs) -> None:
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)

            single = image_processing(image_inputs[0], return_tensors="pt")
            self.assertEqual(
                tuple(single.pixel_values.shape),
                self.image_processor_tester.expected_output_image_shape([image_inputs[0]]),
            )
            self.assertEqual(tuple(single.image_grid_hw.shape), (1, 2))

            batched = image_processing(image_inputs, return_tensors="pt")
            self.assertEqual(
                tuple(batched.pixel_values.shape),
                self.image_processor_tester.expected_output_image_shape(image_inputs),
            )
            self.assertEqual(tuple(batched.image_grid_hw.shape), (len(image_inputs), 2))
            # The flat table must be the per-image tables concatenated in batch order.
            self.assertEqual(int(batched.image_grid_hw.prod(dim=-1).sum()), batched.pixel_values.shape[0])

    def test_call_pil(self):
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)
        self._check_call(image_inputs)

    def test_call_numpy(self):
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)
        self._check_call(image_inputs)

    def test_call_pytorch(self):
        import torch

        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)
        self._check_call(image_inputs)

    @unittest.skip(reason="NeoMME is RGB-only: a 4-channel input is converted, so the patch width is always 3 * p^2")
    def test_call_numpy_4_channels(self):
        pass

    # --- NeoMME-specific behaviour, kept from the hand-written suite ---

    def make_image(self, height: int, width: int) -> "Image.Image":
        rng = np.random.default_rng(0)
        return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))

    def test_the_two_backends_agree_once_a_resolution_budget_resizes(self):
        """The inherited `test_backends_equivalence` runs with default kwargs, where nothing is resized and
        the two backends come out bit-identical. Resampling is the only place they can diverge, and only PIL
        matches the reference implementation, so each budget is pinned here: antialiased torchvision lands
        within one 8-bit level (`1/127.5` after the rescale), where antialiasing off is off by over a hundred.
        """
        image = self.make_image(64, 39)
        one_level = 1 / 127.5
        patch_size = self.image_processor_tester.patch_size

        for budget in ({"max_side": 16}, {"max_pixels": 24 * 24}, {"min_pixels": 128 * 128}):
            with self.subTest(budget=budget):
                fast = NeoMMEImageProcessor(patch_size=patch_size)(images=[image], return_tensors="np", **budget)
                slow = NeoMMEImageProcessorPil(patch_size=patch_size)(images=[image], return_tensors="np", **budget)

                self.assertEqual(fast["image_grid_hw"].tolist(), slow["image_grid_hw"].tolist())
                self.assertEqual(fast["pixel_values"].shape, slow["pixel_values"].shape)
                np.testing.assert_allclose(fast["pixel_values"], slow["pixel_values"], atol=one_level + 1e-6, rtol=0)

    def test_pixels_are_scaled_to_minus_one_one_and_padding_is_minus_one(self):
        """Padding is added to the RAW image, so padded pixels land at exactly -1 after the rescale."""
        patch_size = self.image_processor_tester.patch_size
        image = Image.fromarray(np.full((patch_size, patch_size + 1, 3), 255, dtype=np.uint8))
        outputs = NeoMMEImageProcessor(patch_size=patch_size)(images=[image], return_tensors="np")

        self.assertEqual(outputs["image_grid_hw"].tolist(), [[1, 2]])
        np.testing.assert_allclose(outputs["pixel_values"][0], np.full(3 * patch_size**2, 1.0), atol=1e-6)
        self.assertAlmostEqual(float(outputs["pixel_values"][1].min()), -1.0, places=6)

    def test_patches_are_row_major_with_pixel_channel_last_layout(self):
        patch_size = self.image_processor_tester.patch_size
        height, width = 2 * patch_size, 2 * patch_size
        array = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
        patches = NeoMMEImageProcessor(patch_size=patch_size)(images=[Image.fromarray(array)], return_tensors="np")[
            "pixel_values"
        ]

        self.assertEqual(patches.shape, (4, 3 * patch_size**2))
        for patch_index, (row, column) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            block = array[row * patch_size : (row + 1) * patch_size, column * patch_size : (column + 1) * patch_size]
            np.testing.assert_allclose(patches[patch_index], block.reshape(-1) / 127.5 - 1.0, atol=1e-6)

    def test_no_resize_by_default_and_the_budgets_bound_the_grid(self):
        patch_size = self.image_processor_tester.patch_size
        image = self.make_image(64, 32)
        processor = NeoMMEImageProcessor(patch_size=patch_size)

        self.assertEqual(processor(images=[image], return_tensors="np")["image_grid_hw"].tolist(), [[16, 8]])
        capped = processor(images=[image], max_side=16, return_tensors="np")
        self.assertEqual(capped["image_grid_hw"].tolist(), [[4, 2]])
        # max_side only ever shrinks; min_pixels is the one setting that grows an image.
        small = self.make_image(patch_size, patch_size)
        self.assertEqual(
            processor(images=[small], max_side=1024, return_tensors="np")["image_grid_hw"].tolist(), [[1, 1]]
        )
        self.assertEqual(
            processor(images=[small], min_pixels=16 * 16, return_tensors="np")["image_grid_hw"].tolist(), [[4, 4]]
        )

    def test_the_caps_clamp_the_min_pixels_floor(self):
        """A cap beats the floor. The floor used to ASSIGN the scale, so setting `min_pixels` next to a cap
        silently discarded the cap and emitted a grid several times over budget."""
        processor = NeoMMEImageProcessor(patch_size=self.image_processor_tester.patch_size)
        image = self.make_image(64, 32)

        for cap in ({"max_side": 16}, {"max_pixels": 64 * 32 // 4}):
            with self.subTest(cap=cap):
                capped = processor(images=[image], return_tensors="np", **cap)["image_grid_hw"].tolist()
                floored = processor(images=[image], min_pixels=10**6, return_tensors="np", **cap)["image_grid_hw"]
                self.assertEqual(floored.tolist(), capped)

        # The case above has a source LARGER than the cap, so a guarded clamp would fire on the source size
        # alone. Only a source small enough to be under the cap until the floor grows it past it pins the
        # clamp: 4x4 grown toward 1024 px would be 32x32, and the 8px cap has to cut it back to 8x8.
        grid = processor(images=[self.make_image(4, 4)], max_side=8, min_pixels=1024, return_tensors="np")
        self.assertEqual(grid["image_grid_hw"].tolist(), [[2, 2]])

    def test_get_number_of_image_patches_matches_the_emitted_grid(self):
        processor = NeoMMEImageProcessor(patch_size=self.image_processor_tester.patch_size)
        for height, width, kwargs in [(9, 13, {}), (64, 32, {"max_side": 16}), (4, 4, {"min_pixels": 256})]:
            outputs = processor(images=[self.make_image(height, width)], return_tensors="np", **kwargs)
            expected = int(np.prod(outputs["image_grid_hw"][0]))
            self.assertEqual(processor.get_number_of_image_patches(height, width, kwargs), expected)
