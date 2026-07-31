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
"""Image processor class for NeoMME."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


def get_resize_scale(
    height: int, width: int, max_side: int | None, max_pixels: int | None, min_pixels: int | None
) -> float:
    """Compute the resize scale before patchifying an image.

    Args:
        height (`int`):
            Image height in pixels.
        width (`int`):
            Image width in pixels.
        max_side (`int`, *optional*):
            Maximum longest side. Downscale only.
        max_pixels (`int`, *optional*):
            Maximum pixel area. Downscale only.
        min_pixels (`int`, *optional*):
            Minimum pixel area. May upscale the image. Caps take precedence when both bounds apply.

    Returns:
        `float`: Scale factor to apply to the image.
    """
    scale = 1.0
    if min_pixels is not None and height * width < min_pixels:
        scale = (min_pixels / (height * width)) ** 0.5
    # `min` does double duty: it keeps a cap from ever upscaling, and it lets a cap override the floor.
    if max_side is not None:
        scale = min(scale, max_side / max(height, width))
    if max_pixels is not None:
        scale = min(scale, (max_pixels / (height * width)) ** 0.5)
    return scale


def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """`(num_channels, height, width)` -> `(num_patches, patch_size * patch_size * num_channels)`, row-major."""
    num_channels, height, width = image.shape
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    patches = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patches = patches.transpose(1, 3, 2, 4, 0)
    return patches.reshape(num_patches_height * num_patches_width, -1)


class NeoMMEImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to `self.patch_size`):
        Side, in pixels, of one patch token. The image is padded to a whole multiple of it.
    max_side (`int`, *optional*):
        Longest-side cap in pixels. Unset means no longest-side resize.
    max_pixels (`int`, *optional*):
        Pixel-area cap. Unset means no area cap.
    min_pixels (`int`, *optional*):
        Pixel-area floor; may upscale the image. Caps take precedence when both bounds apply.
    """

    patch_size: int
    max_side: int | None
    max_pixels: int | None
    min_pixels: int | None


@auto_docstring
class NeoMMEImageProcessorPil(PilBackend):
    r"""
    Constructs a NeoMME image processor.

    Images are converted into a row-major grid of flattened RGB patches. By default, no resizing is
    applied; pass `max_side`, `max_pixels`, or `min_pixels` to bound the patch budget.
    """

    valid_kwargs = NeoMMEImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    # `pixel / 127.5 - 1`: `image_mean` is a shift and `image_std` a no-op (as in chameleon), not dataset
    # statistics. The usual `1/255` + `0.5` form is the same map to ~1e-7; this one is bit-exact.
    image_mean = [1.0, 1.0, 1.0]
    image_std = [1.0, 1.0, 1.0]
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 127.5
    do_normalize = True
    patch_size = 32
    max_side = None
    max_pixels = None
    min_pixels = None
    model_input_names = ["pixel_values", "image_grid_hw"]

    def __init__(self, **kwargs: Unpack[NeoMMEImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[NeoMMEImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _validate_preprocess_kwargs(self, **kwargs) -> tuple:
        # `size` is computed per image from the resolution budget, so the generic `do_resize` check
        # (which insists on a `size`) does not apply.
        kwargs.pop("do_resize", None)
        return super()._validate_preprocess_kwargs(**kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        patch_size: int,
        max_side: int | None,
        max_pixels: int | None,
        min_pixels: int | None,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        pixel_values: list[np.ndarray] = []
        image_grid_hw: list[tuple[int, int]] = []

        for image in images:
            if do_resize:
                image = self._resize_to_budget(image, max_side, max_pixels, min_pixels, resample)
            # Pad to a whole patch grid on the RAW image, so padded pixels rescale to -1 exactly like the
            # black canvas the reference implementation pastes onto.
            image, grid_height, grid_width = self._pad_to_patch_grid(image, patch_size)

            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            pixel_values.append(convert_image_to_patches(image, patch_size))
            image_grid_hw.append((grid_height, grid_width))

        return BatchFeature(
            data={
                "pixel_values": np.concatenate(pixel_values, axis=0),
                "image_grid_hw": np.array(image_grid_hw, dtype=np.int64),
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        """Number of patch tokens one `height x width` image becomes (excluding the row-break tokens)."""
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size") or self.patch_size
        scale = get_resize_scale(
            height,
            width,
            images_kwargs.get("max_side", self.max_side),
            images_kwargs.get("max_pixels", self.max_pixels),
            images_kwargs.get("min_pixels", self.min_pixels),
        )
        height, width = max(1, round(height * scale)), max(1, round(width * scale))
        return -(-height // patch_size) * (-(-width // patch_size))

    def _resize_to_budget(
        self,
        image: np.ndarray,
        max_side: int | None,
        max_pixels: int | None,
        min_pixels: int | None,
        resample: "PILImageResampling | None",
    ) -> np.ndarray:
        height, width = image.shape[-2], image.shape[-1]
        scale = get_resize_scale(height, width, max_side, max_pixels, min_pixels)
        if scale == 1.0:
            return image
        size = SizeDict(height=max(1, round(height * scale)), width=max(1, round(width * scale)))
        return self.resize(image=image, size=size, resample=resample)

    def _pad_to_patch_grid(self, image: np.ndarray, patch_size: int) -> tuple[np.ndarray, int, int]:
        height, width = image.shape[-2], image.shape[-1]
        grid_height, grid_width = -(-height // patch_size), -(-width // patch_size)
        pad_height = grid_height * patch_size - height
        pad_width = grid_width * patch_size - width
        if pad_height or pad_width:
            image = np.pad(image, ((0, 0), (0, pad_height), (0, pad_width)), mode="constant", constant_values=0)
        return image, grid_height, grid_width


__all__ = ["NeoMMEImageProcessorPil"]
