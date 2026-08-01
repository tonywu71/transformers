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
"""Testing suite for the NeoMME processor."""

import tempfile
import unittest

import numpy as np

from parameterized import parameterized

from transformers.testing_utils import require_tokenizers, require_torch, require_vision
from transformers.utils import is_tokenizers_available, is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_tokenizers_available():
    from tokenizers import Tokenizer, models, pre_tokenizers

if is_vision_available():
    from PIL import Image

    from transformers import NeoMMEImageProcessor, NeoMMEProcessor, PreTrainedTokenizerFast

if is_torch_available():
    import torch


@require_torch
@require_vision
@require_tokenizers
class NeoMMEProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = NeoMMEProcessor if is_vision_available() else None
    images_input_name = "pixel_values"
    patch_size = 4
    # Frozen special-token block: each special's id is its index in this list.
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>", "<doc>", "<img>", "<query>", "<row>"]

    @classmethod
    def _setup_tokenizer(cls, specials: list[str] | None = None) -> "PreTrainedTokenizerFast":
        """Whitespace word-level tokenizer with specials at frozen ids (built locally, no Hub)."""
        specials = specials if specials is not None else cls.special_tokens
        vocab_words = ["hello", "world", "a", "document", "query", "text", "lower", "newer"]
        vocabulary = {token: index for index, token in enumerate(specials)}
        for word in vocab_words:
            vocabulary[word] = len(vocabulary)

        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            backend.save(handle.name)
            return PreTrainedTokenizerFast(
                tokenizer_file=handle.name,
                pad_token="<pad>",
                eos_token="<eos>",
                unk_token="<unk>",
                mask_token="<mask>",
                # Only the markers this vocabulary really has: naming one here would add it back.
                additional_special_tokens=[t for t in ("<doc>", "<img>", "<query>", "<row>") if t in vocabulary],
            )

    @classmethod
    def setUpClass(cls):
        """Assemble the processor from local components (staging checkpoint is private)."""
        cls.tmpdirname = tempfile.mkdtemp()
        processor = cls.processor_class(
            image_processor=NeoMMEImageProcessor(patch_size=cls.patch_size), tokenizer=cls._setup_tokenizer()
        )
        cls._setup_test_attributes(processor)
        processor.save_pretrained(cls.tmpdirname)

    @property
    def marker_ids(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.special_tokens)}

    @unittest.skip(reason="NeoMMEProcessor takes exactly one of text or images: they are opposite retrieval sides")
    def test_processor_with_multiple_inputs(self):
        pass

    @unittest.skip(reason="every text gets a marker prefix, so processor output never equals raw tokenizer output")
    def test_tokenizer_defaults(self):
        pass

    @unittest.skip(reason="every text gets a marker prefix, so processor output never equals raw tokenizer output")
    def test_tokenizer_decode_defaults(self):
        pass

    @parameterized.expand([(1, "pt"), (2, "pt")])
    @unittest.skip(reason="the chat template is text-only; the image grid is laid out by the image processor")
    def test_apply_chat_template_image(self, batch_size, return_tensors):
        pass

    def test_tokenizer_defaults_preserved_by_kwargs(self):
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        inputs = processor(text=self.prepare_text_inputs(), return_tensors="pt")
        self.assertEqual(inputs[self.text_input_name].shape[-1], 117)

    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding="longest")
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        inputs = processor(text=self.prepare_text_inputs(), return_tensors="pt", max_length=112, padding="max_length")
        self.assertEqual(inputs[self.text_input_name].shape[-1], 112)

    def test_unstructured_kwargs(self):
        processor = self.processor_class(**self.prepare_components())
        self.skip_processor_without_typed_kwargs(processor)

        inputs = processor(text=self.prepare_text_inputs(), return_tensors="pt", padding="max_length", max_length=76)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_structured_kwargs_nested(self):
        processor = self.processor_class(**self.prepare_components())
        self.skip_processor_without_typed_kwargs(processor)

        inputs = processor(
            text=self.prepare_text_inputs(),
            common_kwargs={"return_tensors": "pt"},
            text_kwargs={"padding": "max_length", "max_length": 76},
        )
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_structured_kwargs_nested_from_dict(self):
        """Same merge path as nested kwargs, but via a single dict of dicts."""
        processor = self.processor_class(**self.prepare_components())
        self.skip_processor_without_typed_kwargs(processor)

        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }
        inputs = processor(text=self.prepare_text_inputs(), **all_kwargs)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_flat_kwarg_applied_when_modality_dict_lacks_it(self):
        """Flat `return_tensors` must survive next to a `text_kwargs` dict that omits it (regression #46192)."""
        processor = self.get_processor()
        self.skip_processor_without_typed_kwargs(processor)

        inputs = processor(text=self.prepare_text_inputs(), text_kwargs={"padding": "longest"}, return_tensors="np")
        self.assertIsInstance(inputs[self.text_input_name], np.ndarray)

    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        """`rescale_factor=-1.0` drives every pixel negative, so a preserved default shows up in the mean."""
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=-1.0
        )
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        inputs = processor(images=self.prepare_image_inputs(), return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_kwargs_overrides_default_image_processor_kwargs(self):
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=1
        )
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        inputs = processor(
            images=self.prepare_image_inputs(), do_rescale=True, rescale_factor=-1.0, return_tensors="pt"
        )
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_unstructured_kwargs_batched(self):
        processor = self.processor_class(**self.prepare_components())
        self.skip_processor_without_typed_kwargs(processor)

        inputs = processor(
            images=self.prepare_image_inputs(batch_size=2), return_tensors="pt", do_rescale=True, rescale_factor=-1.0
        )
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_doubly_passed_kwargs(self):
        processor = self.processor_class(**self.prepare_components())
        self.skip_processor_without_typed_kwargs(processor)

        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            processor(
                images=image_input,
                images_kwargs={"do_rescale": True, "rescale_factor": -1.0},
                do_rescale=True,
                return_tensors="pt",
            )

    def test_model_input_names(self):
        processor = self.get_processor()
        inputs = processor(images=self.prepare_image_inputs())
        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_padding_and_return_tensors(self):
        """Padding and `return_tensors` used to be dropped; only `max_length` survived the merge."""
        processor = self.get_processor()

        padded = processor(text=["hello world", "a"], text_role="document", padding="max_length", max_length=32)
        self.assertEqual(padded["input_ids"].shape, (2, 32))
        self.assertEqual(int(padded["attention_mask"][1].sum()), 2)

        for return_tensors, expected in (("np", np.ndarray), ("pt", torch.Tensor)):
            with self.subTest(return_tensors=return_tensors):
                batch = processor(text=["hello world"], text_role="query", return_tensors=return_tensors)
                self.assertIsInstance(batch["input_ids"], expected)

    def test_unsupported_text_kwargs_raise(self):
        processor = self.get_processor()

        with self.assertRaises(ValueError):  # would corrupt the marker layout
            processor(text=["hello"], add_special_tokens=True)
        with self.assertRaises(ValueError):  # a max_length always truncates the content
            processor(text=["hello world text"], max_length=4, truncation=False)
        with self.assertRaises(ValueError):  # ragged rows cannot become one tensor
            processor(text=["hello world", "a"], padding=False)
        with self.assertRaises(ValueError):  # nothing to pad to
            processor(text=["hello"], padding="max_length")

    def test_tokenizer_init_padding_side(self):
        """`tokenizer.init_kwargs` padding_side must not be treated as a caller kwarg and refuse every text call."""
        processor = self.get_processor()

        for side in ("right", "left"):
            processor.tokenizer.init_kwargs["padding_side"] = side
            self.assertEqual(processor(text=["hello"], text_role="query")["input_ids"].shape[0], 1)

        processor.tokenizer.init_kwargs.pop("padding_side", None)
        # Asking for it explicitly still raises: this processor always right-pads.
        with self.assertRaises(ValueError):
            processor(text=["hello"], padding_side="left")
        with self.assertRaises(ValueError):
            processor(text=["hello"], text_kwargs={"padding_side": "left"})

    def test_query_marker_and_expansion(self):
        processor = self.get_processor()
        batch = processor(text=["hello world", "a"], text_role="query")
        first = batch["input_ids"][0].tolist()

        self.assertEqual(first[0], self.marker_ids["<query>"])
        self.assertEqual(first[-processor.query_expand :], [self.marker_ids["<mask>"]] * processor.query_expand)
        self.assertEqual(len(first), 1 + 2 + processor.query_expand)
        # The shorter query is right-padded and its padding is masked out.
        self.assertEqual(int(batch["attention_mask"][1].sum()), 1 + 1 + processor.query_expand)

    def test_query_truncation_preserves_markers(self):
        processor = self.get_processor()
        max_length = 1 + 1 + processor.query_expand
        ids = processor(text=["hello world text"], text_role="query", max_length=max_length)["input_ids"][0].tolist()

        self.assertEqual(len(ids), max_length)
        self.assertEqual(ids[0], self.marker_ids["<query>"])
        self.assertEqual(ids[-processor.query_expand :], [self.marker_ids["<mask>"]] * processor.query_expand)

    def test_query_expansion_fits_max_length(self):
        processor = self.get_processor()
        with self.assertRaises(ValueError):
            processor(text=["hello"], text_role="query", max_length=processor.query_expand)

    def test_document_marker(self):
        processor = self.get_processor()
        batch = processor(text=["hello world", ""], text_role="document")
        first = batch["input_ids"][0].tolist()

        self.assertEqual(first[0], self.marker_ids["<doc>"])
        self.assertNotIn(self.marker_ids["<mask>"], first)
        # An empty string becomes a single space, so no document is marker-only.
        self.assertEqual(int(batch["attention_mask"][1].sum()), 1)

    def test_one_modality_per_call(self):
        processor = self.get_processor()
        image = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            processor()
        with self.assertRaises(ValueError):
            processor(text=["hello"], images=[image])

    def test_missing_markers_raise(self):
        """A missing marker used to resolve to `unk_token_id` and silently open every document with `<unk>`."""
        stripped = self._setup_tokenizer(specials=[token for token in self.special_tokens if token != "<row>"])
        processor = NeoMMEProcessor(
            image_processor=NeoMMEImageProcessor(patch_size=self.patch_size), tokenizer=stripped
        )

        with self.assertRaises(ValueError) as raised:
            processor(text=["hello world"], text_role="document")
        self.assertIn("row", str(raised.exception))

    def test_image_layout(self):
        processor = self.get_processor()
        grid_height, grid_width = 2, 3
        patch_size = self.patch_size
        image = Image.fromarray(
            np.random.randint(0, 255, (grid_height * patch_size, grid_width * patch_size, 3), dtype=np.uint8)
        )
        batch = processor(images=[image])
        ids = batch["input_ids"][0].tolist()
        positions = batch["position_ids"][:, 0]

        expected = [self.marker_ids["<doc>"], self.marker_ids["<img>"]]
        for _ in range(grid_height):
            expected += [self.marker_ids["<img>"]] * grid_width + [self.marker_ids["<row>"]]
        self.assertEqual(ids, expected)
        self.assertEqual(batch["pixel_values"].shape, (grid_height * grid_width, 3 * patch_size**2))
        self.assertEqual(batch["image_grid_hw"].tolist(), [[grid_height, grid_width]])

        # The two markers take diagonal positions, then the grid starts at (2, 2).
        self.assertEqual(positions[:, 0].tolist(), [0, 0])
        self.assertEqual(positions[:, 1].tolist(), [1, 1])
        self.assertEqual(positions[:, 2].tolist(), [2, 2])
        self.assertEqual(positions[:, 2 + grid_width].tolist(), [2, 2 + grid_width])
        self.assertEqual(positions[:, 2 + grid_width + 1].tolist(), [3, 2])

    def test_per_image_position_ids(self):
        processor = self.get_processor()
        patch_size = self.patch_size
        images = [
            Image.fromarray(np.random.randint(0, 255, (2 * patch_size, 3 * patch_size, 3), dtype=np.uint8)),
            Image.fromarray(np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)),
        ]
        batch = processor(images=images)

        # A running offset across the batch would shift every image after the first.
        self.assertEqual(batch["position_ids"][:, 1, 0].tolist(), [0, 0])
        self.assertEqual(batch["position_ids"][:, 1, 1].tolist(), [1, 1])
        self.assertEqual(batch["pixel_values"].shape[0], 2 * 3 + 1)
        self.assertEqual(int(batch["attention_mask"][1].sum()), 2 + 1 * (1 + 1))

    def test_score_retrieval(self):
        processor = self.get_processor()

        with self.subTest(mode="maxsim"):
            query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
            passages = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])
            scores = processor.score_retrieval(query, passages)
            self.assertEqual(scores.shape, (1, 2))
            # Passage 0 has a padding token (a zero row); its max must ignore it.
            torch.testing.assert_close(scores[0], torch.tensor([0.5, 0.5]))

        with self.subTest(mode="maxsim_normalize"):
            query = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
            passage = torch.tensor([[[1.0, 0.0]]])
            torch.testing.assert_close(processor.score_retrieval(query, passage)[0], torch.tensor([1.0]))
            torch.testing.assert_close(
                processor.score_retrieval(query, passage, normalize=False)[0], torch.tensor([2.0])
            )

        with self.subTest(mode="maxsim_empty_passage"):
            query = torch.tensor([[[1.0, 0.0]]])
            passages = torch.tensor([[[1.0, 0.0]], [[0.0, 0.0]]])
            torch.testing.assert_close(processor.score_retrieval(query, passages)[0], torch.tensor([1.0, -1.0]))

        with self.subTest(mode="dense_cosine"):
            queries = torch.tensor([[1.0, 0.0]])
            passages = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
            torch.testing.assert_close(processor.score_retrieval(queries, passages)[0], torch.tensor([1.0, 0.0]))
