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
"""Testing suite for the PyTorch NeoMME model."""

import os
import unittest
from typing import ClassVar
from unittest.mock import patch

from datasets import load_dataset

from transformers import NeoMMEConfig, is_torch_available
from transformers.testing_utils import cleanup, require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import NeoMMEForMaskedLM, NeoMMEForRetrieval, NeoMMEModel, NeoMMEProcessor
    from transformers import initialization as init
    from transformers.models.neomme.modeling_neomme import (
        NeoMMEAttention,
        NeoMMEMLP,
        NeoMMEPreTrainedModel,
        NeoMMEValueEmbeddings,
    )


def _give_the_residual_branches_weight(test_case: unittest.TestCase) -> None:
    """Patch `_init_weights` for the duration of `test_case` so a freshly built model is not an identity.

    NeoMME starts every residual branch at exactly zero — `o_proj`, `down_proj`, the XSA `alpha` and the
    value-embedding table — mirroring the research init. A model built from the config alone is therefore a
    bitwise identity on the embedding stream: scrambling `q_proj`, `kv_proj`, `up_proj`, `output_gate` and
    `lambdas` leaves `last_hidden_state` unchanged to the last bit. Every inherited test that compares
    outputs (eager vs sdpa, padded vs unpadded batch, flash-attention equivalence, torch.compile) would
    then pass whatever the trunk computes. Standard fan-in init on those four tensors is enough to make the
    comparisons bite. The `init.*` helpers skip anything a checkpoint already carries, so this only ever
    touches weights that would otherwise be born zero.
    """
    initialize = NeoMMEPreTrainedModel._init_weights

    @torch.no_grad()
    def initialize_with_live_residual_branches(self: NeoMMEPreTrainedModel, module: torch.nn.Module) -> None:
        initialize(self, module)
        if isinstance(module, NeoMMEAttention):
            init.normal_(module.o_proj.weight, mean=0.0, std=module.o_proj.weight.shape[-1] ** -0.5)
            if module.alpha is not None:
                # XSA scales by `tanh(alpha)`, so a small std would leave it a no-op just like zero does.
                init.normal_(module.alpha, mean=0.0, std=1.0)
        elif isinstance(module, NeoMMEMLP):
            init.normal_(module.down_proj.weight, mean=0.0, std=module.down_proj.weight.shape[-1] ** -0.5)
        elif isinstance(module, NeoMMEValueEmbeddings):
            init.normal_(module.weight, mean=0.0, std=module.weight.shape[-1] ** -0.5)

    patcher = patch.object(NeoMMEPreTrainedModel, "_init_weights", initialize_with_live_residual_branches)
    patcher.start()
    test_case.addCleanup(patcher.stop)


class NeoMMEModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=13,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        embedding_rank=16,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        # 16, not 8: the default `partial_rotary_factor` of 0.25 has to leave a multiple of 4 rotating dims
        # on full-attention layers (4 here), which is also the smallest width that exercises both M-RoPE axes.
        head_dim=16,
        global_attn_every_n_layers=3,
        sliding_window_short=3,
        sliding_window_long=6,
        patch_size=4,
        embedding_dim=8,
        max_position_embeddings=128,
        initializer_range=0.02,
        pad_token_id=0,
        mask_token_id=4,
        document_token_id=5,
        image_token_id=6,
        query_token_id=7,
        row_token_id=8,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.embedding_rank = embedding_rank
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.sliding_window_short = sliding_window_short
        self.sliding_window_long = sliding_window_long
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.document_token_id = document_token_id
        self.image_token_id = image_token_id
        self.query_token_id = query_token_id
        self.row_token_id = row_token_id

    def get_config(self):
        config = NeoMMEConfig(
            vocab_size=self.vocab_size,
            embedding_rank=self.embedding_rank,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            global_attn_every_n_layers=self.global_attn_every_n_layers,
            sliding_window_short=self.sliding_window_short,
            sliding_window_long=self.sliding_window_long,
            patch_size=self.patch_size,
            embedding_dim=self.embedding_dim,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            document_token_id=self.document_token_id,
            image_token_id=self.image_token_id,
        )
        if test := os.environ.get("PYTEST_CURRENT_TEST", None):
            test_name = test.split(":")[-1].split(" ")[0]
            # Only the eager attention path can return attention probabilities.
            if test_name in (
                "test_attention_outputs",
                "test_hidden_states_output",
                "test_retain_grad_hidden_states_attentions",
            ):
                config._attn_implementation = "eager"
        return config

    def prepare_config_and_inputs(self):
        # Keep the ids clear of the frozen special-token block so no text token doubles as a marker.
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 64) + 64
        input_mask = random_attention_mask([self.batch_size, self.seq_length]) if self.use_input_mask else None
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size) if self.use_labels else None
        return self.get_config(), input_ids, input_mask, token_labels

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask, _ = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": input_mask}

    def prepare_image_config_and_inputs(self, grid_height=2, grid_width=3):
        """One image per row, laid out exactly as [`NeoMMEProcessor`] emits it."""
        config = self.get_config()
        sequence = [config.document_token_id, config.image_token_id]
        for _ in range(grid_height):
            sequence += [config.image_token_id] * grid_width + [self.row_token_id]
        input_ids = torch.tensor([sequence] * self.batch_size)
        pixel_values = floats_tensor([self.batch_size * grid_height * grid_width, config.patch_dim])
        return config, input_ids, pixel_values

    def create_and_check_model(self, config, input_ids, input_mask, token_labels):
        model = NeoMMEModel(config=config).to(torch_device).eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertTrue(torch.isfinite(result.last_hidden_state).all())

    def create_and_check_for_masked_lm(self, config, input_ids, input_mask, token_labels):
        model = NeoMMEForMaskedLM(config=config).to(torch_device).eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertTrue(torch.isfinite(result.loss))

    def create_and_check_for_retrieval(self, config, input_ids, input_mask, token_labels):
        model = NeoMMEForRetrieval(config=config).to(torch_device).eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(
            result.multivector_embeddings.shape, (self.batch_size, self.seq_length, self.embedding_dim)
        )
        self.parent.assertEqual(result.dense_embeddings.shape, (self.batch_size, self.hidden_size))


@require_torch
class NeoMMEModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (NeoMMEModel, NeoMMEForMaskedLM) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    # The common batch is text-only, so the vision stem legitimately receives no gradient. The dedicated
    # `test_patch_stem_receives_gradients_from_images` covers it instead.
    test_all_params_have_gradient = False
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = NeoMMEModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NeoMMEConfig)
        _give_the_residual_branches_weight(self)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    def test_for_masked_lm(self):
        self.model_tester.create_and_check_for_masked_lm(*self.model_tester.prepare_config_and_inputs())

    @unittest.skip(
        reason="value embeddings are a second vocab-indexed table read inside attention, so an "
        "`inputs_embeds` forward has no ids to look them up with and deliberately omits them"
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(
        reason="the per-layer-type `ntk_inv_freq <= original_inv_freq` check compares a layer type the test "
        "never forwards, so both sides are init-time values: NeoMME's default RoPE is written `theta ** -x` "
        "to stay bit-identical to the research implementation every checkpoint was trained with, and upstream's "
        "dynamic init uses `1.0 / theta ** x`, which is 1 ULP larger on some frequencies. Adopting upstream's "
        "form instead would shift cos/sin by 6.1e-5 at the end of the context, 61% of the conversion parity "
        "budget, for no behavioural gain. The three `test_model_rope_scaling_from_config` variants, which check "
        "that scaling actually changes the output, do run."
    )
    def test_model_rope_scaling_frequencies(self):
        pass

    def test_the_trunk_is_not_an_identity_at_init(self):
        """Guard on `_give_the_residual_branches_weight`: without it every output comparison here is
        vacuous, because the zero-initialised `o_proj` / `down_proj` make the whole trunk a no-op."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = NeoMMEModel(config).to(torch_device).eval()
        inputs = {"input_ids": inputs_dict["input_ids"], "attention_mask": inputs_dict["attention_mask"]}

        with torch.no_grad():
            before = model(**inputs).last_hidden_state.clone()
            model.layers[0].self_attn.q_proj.weight.normal_(mean=0.0, std=1.0)
            after = model(**inputs).last_hidden_state

        self.assertGreater((after - before).abs().max().item(), 1e-4)

    def test_sdpa_can_dispatch_on_flash(self):
        """Not reachable: every layer is handed a 4-D mask, and torch's flash backend refuses any mask.

        Sliding layers need their band expressed as a mask, and SDPA's flash kernel only understands
        `is_causal` or no mask at all ("Flash Attention does not support non-null attn_mask"). The real
        flash path for a windowed bidirectional model is the flash-attention package, which takes
        `window_size` directly and is covered by `test_flash_attn_2_inference_equivalence`.
        """
        self.skipTest(reason="every NeoMME layer passes a 4-D mask; SDPA's flash kernel rejects masks")

    def test_layer_pattern_makes_the_last_layer_global(self):
        config = self.model_tester.get_config()
        expected = [
            "full_attention"
            if (i + 1) % config.global_attn_every_n_layers == 0 or i == config.num_hidden_layers - 1
            else "sliding_attention"
            for i in range(config.num_hidden_layers)
        ]
        self.assertEqual(config.layer_types, expected)
        self.assertEqual(config.layer_types[-1], "full_attention")

    def test_layer_types_disagreeing_with_the_stride_raises(self):
        """`layer_types` is what gets serialized, so it must never silently contradict the stride."""
        base = {"num_hidden_layers": 4, "global_attn_every_n_layers": 3}
        with self.assertRaises(ValueError):
            NeoMMEConfig(**base, layer_types=["sliding_attention"] * 4)
        with self.assertRaises(ValueError):  # one entry short
            NeoMMEConfig(**base, layer_types=["sliding_attention"] * 2 + ["full_attention"])
        with self.assertRaises(ValueError):  # not a known layer type
            NeoMMEConfig(**base, layer_types=["sliding_attention", "gdn", "full_attention", "full_attention"])

        # A hand-written pattern no stride can produce is legal once the stride is disowned.
        pattern = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
        config = NeoMMEConfig(num_hidden_layers=4, global_attn_every_n_layers=None, layer_types=pattern)
        self.assertEqual(config.layer_types, pattern)

    def test_the_fixed_architecture_switches_are_refused_not_ignored(self):
        """`use_xsa`, `depth_scale`, `patch_stem` and `cheap_mixer` are research ablation switches, and the
        port implements exactly one value of each — the one every released NeoMME was trained with. A
        config that names another must fail loudly: `depth_scale` in particular changes no tensor shape, so
        a checkpoint trained without it would otherwise load clean and be quietly wrong."""
        base = {"num_hidden_layers": 4, "global_attn_every_n_layers": 3}
        for name, unsupported in (
            ("use_xsa", False),
            ("depth_scale", False),
            ("patch_stem", "linear"),
            ("cheap_mixer", "gdn"),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                NeoMMEConfig(**base, **{name: unsupported})

        # The released values are accepted, and leave no trace in `config.json`.
        config = NeoMMEConfig(**base, use_xsa=True, depth_scale=True, patch_stem="mlp", cheap_mixer="swa")
        self.assertEqual([key for key in NeoMMEConfig.fixed_architecture if key in config.to_dict()], [])

    def test_the_two_window_widths_are_validated(self):
        """One band width is two equal widths, not a magic `sliding_window_long = 0`. The research config
        uses zero for 'uniform', and a zero half-width here would mean a diagonal-only band."""
        base = {"num_hidden_layers": 4, "global_attn_every_n_layers": 3}
        uniform = NeoMMEConfig(**base, sliding_window_short=256, sliding_window_long=256)
        # Stride 3 over 4 layers leaves two sliding layers, which would alternate short/long.
        self.assertEqual([w for w in uniform.layer_window_sizes if w is not None], [256, 256])
        for short, long in ((256, 0), (256, 128), (0, 256)):
            with self.subTest(short=short, long=long), self.assertRaises(ValueError):
                NeoMMEConfig(**base, sliding_window_short=short, sliding_window_long=long)

    def test_rope_parameters_follow_layer_types(self):
        """A homogeneous pattern is legal, and only the layer types in use get rope parameters.

        Keying rope on a layer type the model does not have sends `standardize_rope_params` down its
        single-global-dict branch, which writes flat keys the annotation then rejects.
        """
        self.assertEqual(list(NeoMMEConfig(num_hidden_layers=1).rope_parameters), ["full_attention"])
        self.assertEqual(list(NeoMMEConfig(global_attn_every_n_layers=1).rope_parameters), ["full_attention"])
        sliding_only = NeoMMEConfig(
            num_hidden_layers=4, global_attn_every_n_layers=None, layer_types=["sliding_attention"] * 4
        )
        self.assertEqual(list(sliding_only.rope_parameters), ["sliding_attention"])

    def test_a_flat_rope_theta_reaches_every_layer_type(self):
        """`rope_theta` is the standard knob, so it must not be dropped in favour of the defaults, nor
        survive into `config.json` as a field nothing reads."""
        config = NeoMMEConfig(rope_theta=123456.0)
        self.assertEqual(
            {layer_type: params["rope_theta"] for layer_type, params in config.rope_parameters.items()},
            {"full_attention": 123456.0, "sliding_attention": 123456.0},
        )
        self.assertNotIn("rope_theta", config.to_dict())
        # An explicit per-layer-type value still wins over the flat one.
        explicit = NeoMMEConfig(rope_theta=123456.0, rope_parameters={"sliding_attention": {"rope_theta": 7.0}})
        self.assertEqual(explicit.rope_parameters["sliding_attention"]["rope_theta"], 7.0)
        self.assertEqual(explicit.rope_parameters["full_attention"]["rope_theta"], 123456.0)

    def test_a_partial_rotary_factor_that_does_not_divide_by_four_is_refused(self):
        """The two M-RoPE axes take alternating frequency pairs, so the rotating dims divide by 2 twice.
        This used to be rounded down inside the model, which built a working model whose spectrum was
        narrower than its config said — at `head_dim=8` with the default 0.25, silently zero rotary dims."""
        with self.assertRaisesRegex(ValueError, "not a multiple of 4"):
            NeoMMEConfig(head_dim=8)
        with self.assertRaisesRegex(ValueError, "not a multiple of 4"):
            NeoMMEConfig(head_dim=64, rope_parameters={"full_attention": {"partial_rotary_factor": 0.3}})
        # A factor that lands on a multiple of 4 is fine, on either layer type.
        config = NeoMMEConfig(head_dim=64, rope_parameters={"full_attention": {"partial_rotary_factor": 0.75}})
        self.assertEqual(config.rope_parameters["full_attention"]["partial_rotary_factor"], 0.75)

    def test_config_round_trips_through_a_dict(self):
        config = self.model_tester.get_config()
        reloaded = NeoMMEConfig.from_dict(config.to_dict())

        self.assertEqual(reloaded.layer_types, config.layer_types)
        self.assertEqual(reloaded.layer_window_sizes, config.layer_window_sizes)
        self.assertEqual(reloaded.rope_parameters, config.rope_parameters)

    def test_sliding_windows_alternate_by_sliding_layer_ordinal(self):
        config = self.model_tester.get_config()
        windows = [window for window in config.layer_window_sizes if window is not None]
        expected = [
            config.sliding_window_long if index % 2 else config.sliding_window_short for index in range(len(windows))
        ]
        self.assertEqual(windows, expected)
        self.assertEqual(
            [window is None for window in config.layer_window_sizes],
            [layer_type == "full_attention" for layer_type in config.layer_types],
        )

    def test_every_layer_attends_bidirectionally_within_its_window(self):
        """No layer may be causal, and each sliding layer's band must be exactly its half-width.

        Post-softmax weights are exactly 0 where the mask forbids attention, so this reads the pattern
        straight off `output_attentions`: the upper triangle must be populated (nothing causal), and a
        sliding layer must be zero outside `abs(i - j) <= window`.
        """
        config = self.model_tester.get_config()
        config._attn_implementation = "eager"  # only the eager path returns attention probabilities
        model = NeoMMEModel(config).to(torch_device).eval()

        seq_length = self.model_tester.seq_length
        input_ids = ids_tensor([1, seq_length], config.vocab_size - 64) + 64
        with torch.no_grad():
            attentions = model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), output_attentions=True
            ).attentions

        self.assertEqual(len(attentions), config.num_hidden_layers)
        positions = torch.arange(seq_length, device=torch_device)
        distance = (positions[:, None] - positions[None, :]).abs()

        for layer_idx, (attention, window) in enumerate(zip(attentions, config.layer_window_sizes)):
            inside = distance <= window if window is not None else torch.ones_like(distance, dtype=torch.bool)
            with self.subTest(layer=layer_idx, window=window):
                self.assertTrue((attention[0, :, inside] > 0).all(), "a reachable pair got zero weight")
                if window is not None and (~inside).any():
                    self.assertTrue((attention[0, :, ~inside] == 0).all(), "attention leaked outside the band")
                # The upper triangle carries the bidirectionality: a causal mask would zero it.
                upper = torch.triu(inside, diagonal=1)
                if upper.any():
                    self.assertTrue((attention[0, :, upper] > 0).all(), "layer is causal")

    def test_several_images_in_one_sequence_scatter_in_order(self):
        """Two `<doc> <img>` grids in a single row: patches must land in reading order, none skipped."""
        config = self.model_tester.get_config()
        model = NeoMMEModel(config).to(torch_device).eval()

        grids = [(2, 3), (1, 2)]
        sequence: list[int] = []
        patch_positions: list[int] = []
        for grid_height, grid_width in grids:
            sequence += [config.document_token_id, config.image_token_id]
            for _ in range(grid_height):
                patch_positions += [len(sequence) + offset for offset in range(grid_width)]
                sequence += [config.image_token_id] * grid_width + [self.model_tester.row_token_id]
        input_ids = torch.tensor([sequence], device=torch_device)
        pixel_values = floats_tensor([len(patch_positions), config.patch_dim]).to(torch_device)

        inputs_embeds = model.embeddings(input_ids=input_ids)
        scattered = model._scatter_patch_embeddings(input_ids, inputs_embeds, pixel_values)
        expected = model.patch_embeddings(pixel_values)

        self.assertEqual(len(patch_positions), sum(h * w for h, w in grids))
        # Every placeholder holds ITS patch: a scatter that dropped the second image, or ran the two grids
        # out of order, fails here rather than silently producing plausible embeddings.
        torch.testing.assert_close(scattered[0, patch_positions], expected)
        untouched = [i for i in range(len(sequence)) if i not in patch_positions]
        torch.testing.assert_close(scattered[0, untouched], inputs_embeds[0, untouched])

    def test_the_image_path_compiles_with_a_full_graph(self):
        """For a document retriever the image path is the path, and it used to break `fullgraph=True`:
        counting the patch placeholders reads a value only the runtime knows."""
        config = self.model_tester.get_config()
        config._attn_implementation = "sdpa"
        model = NeoMMEModel(config).to(torch_device).eval()

        grid_width = 3
        sequence = [config.document_token_id, config.image_token_id]
        sequence += [config.image_token_id] * grid_width + [self.model_tester.row_token_id]
        input_ids = torch.tensor([sequence], device=torch_device)
        pixel_values = floats_tensor([grid_width, config.patch_dim]).to(torch_device)

        with torch.no_grad():
            compiled = torch.compile(model, fullgraph=True)(input_ids=input_ids, pixel_values=pixel_values)
            eager = model(input_ids=input_ids, pixel_values=pixel_values)

        torch.testing.assert_close(compiled.last_hidden_state, eager.last_hidden_state)

    def test_value_embeddings_feed_first_and_last_global_layers(self):
        config = self.model_tester.get_config()
        model = NeoMMEModel(config)
        globals_ = [i for i, layer_type in enumerate(config.layer_types) if layer_type == "full_attention"]
        self.assertEqual(model.value_embedding_layers, {globals_[0], globals_[-1]})

    def test_backbone_owns_no_norm_weights(self):
        """Every backbone norm is parameter-free, so the state dict must contain no norm weight but the stem's."""
        config = self.model_tester.get_config()
        norm_keys = [key for key in NeoMMEModel(config).state_dict() if "norm" in key]
        self.assertEqual(sorted(norm_keys), ["patch_embeddings.norm.bias", "patch_embeddings.norm.weight"])

    def test_masked_lm_adds_no_parameters(self):
        config = self.model_tester.get_config()
        self.assertEqual(NeoMMEForMaskedLM(config).num_parameters(), NeoMMEModel(config).num_parameters())
        self.assertIsNone(NeoMMEForMaskedLM(config).get_output_embeddings())

    def test_rotary_embedding_is_interleaved_and_partial(self):
        """The rotation must act on interleaved pairs and leave the NoPE tail untouched."""
        from transformers.models.neomme.modeling_neomme import apply_interleaved_rotary_pos_emb

        head_dim, rotary_dim = 8, 4
        states = torch.arange(head_dim, dtype=torch.float32).view(1, 1, 1, head_dim)
        # A quarter turn on both axes: cos = 0, sin = 1 -> (x0, x1) becomes (-x1, x0).
        cos = torch.zeros(1, 1, rotary_dim // 2)
        sin = torch.ones(1, 1, rotary_dim // 2)
        rotated = apply_interleaved_rotary_pos_emb(states, cos, sin, rotary_dim)
        torch.testing.assert_close(rotated.flatten(), torch.tensor([-1.0, 0.0, -3.0, 2.0, 4.0, 5.0, 6.0, 7.0]))

    def test_image_patches_are_scattered_into_the_grid_placeholders(self):
        """The `<img>` marker right after `<doc>` must not consume a patch embedding."""
        config, input_ids, pixel_values = self.model_tester.prepare_image_config_and_inputs()
        model = NeoMMEModel(config).to(torch_device).eval()
        input_ids, pixel_values = input_ids.to(torch_device), pixel_values.to(torch_device)

        output = model(input_ids=input_ids, pixel_values=pixel_values)
        self.assertEqual(output.last_hidden_state.shape[1], input_ids.shape[1])
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

        with self.assertRaises(ValueError):
            model(input_ids=input_ids, pixel_values=pixel_values[:-1])
        with self.assertRaises(ValueError):
            model(input_ids=input_ids, pixel_values=pixel_values[:, :-1])

    def test_patch_stem_receives_gradients_from_images(self):
        config, input_ids, pixel_values = self.model_tester.prepare_image_config_and_inputs()
        model = NeoMMEForMaskedLM(config).to(torch_device).train()
        input_ids, pixel_values = input_ids.to(torch_device), pixel_values.to(torch_device)

        model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids).loss.backward()
        for name, parameter in model.model.patch_embeddings.named_parameters():
            self.assertIsNotNone(parameter.grad, f"patch_embeddings.{name} received no gradient")
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_short_row_in_a_long_batch_stays_finite(self):
        """The sliding band intersected with padding can leave a padding query row with no key."""
        config = self.model_tester.get_config()
        model = NeoMMEModel(config).to(torch_device).eval()
        seq_length = 4 * config.sliding_window_long
        input_ids = ids_tensor([2, seq_length], config.vocab_size - 64) + 64
        attention_mask = torch.ones_like(input_ids)
        attention_mask[1, 2:] = 0

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

    def test_two_axis_position_ids_match_the_expanded_one_axis_form(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEModel(config).to(torch_device).eval()
        one_axis = torch.arange(input_ids.shape[1], device=torch_device).expand(input_ids.shape[0], -1)

        with torch.no_grad():
            default = model(input_ids=input_ids, attention_mask=input_mask).last_hidden_state
            explicit = model(input_ids=input_ids, attention_mask=input_mask, position_ids=one_axis).last_hidden_state
            stacked = model(
                input_ids=input_ids, attention_mask=input_mask, position_ids=torch.stack([one_axis, one_axis])
            ).last_hidden_state

        torch.testing.assert_close(default, explicit)
        torch.testing.assert_close(default, stacked)


@require_torch
class NeoMMEForRetrievalModelTest(ModelTesterMixin, unittest.TestCase):
    """`NeoMMEForRetrieval` produces embeddings rather than a loss, so it is tested on its own."""

    all_model_classes = (NeoMMEForRetrieval,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_all_params_have_gradient = False

    def setUp(self):
        self.model_tester = NeoMMEModelTester(self, is_training=False)
        self.config_tester = ConfigTester(self, config_class=NeoMMEConfig)
        _give_the_residual_branches_weight(self)

    def test_sdpa_can_dispatch_on_flash(self):
        """Not reachable: every layer is handed a 4-D mask, and torch's flash backend refuses any mask.

        Sliding layers need their band expressed as a mask, and SDPA's flash kernel only understands
        `is_causal` or no mask at all ("Flash Attention does not support non-null attn_mask"). The real
        flash path for a windowed bidirectional model is the flash-attention package, which takes
        `window_size` directly and is covered by `test_flash_attn_2_inference_equivalence`.
        """
        self.skipTest(reason="every NeoMME layer passes a 4-D mask; SDPA's flash kernel rejects masks")

    def test_for_retrieval(self):
        self.model_tester.create_and_check_for_retrieval(*self.model_tester.prepare_config_and_inputs())

    def test_multivector_head_zeroes_padding_and_normalizes(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        input_mask[0, 3:] = 0
        model = NeoMMEForRetrieval(config).to(torch_device).eval()
        embeddings = model(input_ids=input_ids, attention_mask=input_mask).multivector_embeddings

        self.assertTrue((embeddings[0, 3:] == 0).all())
        real = input_mask.bool()
        torch.testing.assert_close(
            embeddings[real].norm(dim=-1), torch.ones_like(embeddings[real][:, 0]), rtol=1e-4, atol=1e-4
        )

    def test_dense_head_truncates_before_normalizing(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEForRetrieval(config).to(torch_device).eval()
        full = model(input_ids=input_ids, attention_mask=input_mask).dense_embeddings
        truncated = model(input_ids=input_ids, attention_mask=input_mask, dense_dim=8).dense_embeddings

        self.assertEqual(truncated.shape[-1], 8)
        torch.testing.assert_close(truncated.norm(dim=-1), torch.ones_like(truncated[:, 0]), rtol=1e-4, atol=1e-4)
        # Truncating a unit vector's prefix and renormalizing is NOT the same as slicing the full vector.
        self.assertFalse(torch.allclose(truncated, full[:, :8], atol=1e-3))

    def test_dense_dim_is_validated_and_the_helpers_take_inputs_first(self):
        """A bad Matryoshka width used to slice silently: `dense_dim=-4` returned a shorter vector that
        downstream cosine scoring cannot tell from a good one."""
        config, input_ids, _, _ = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEForRetrieval(config).to(torch_device).eval()

        # Both helpers take the model inputs in the same order, so a positional call is unambiguous.
        self.assertEqual(
            model.get_dense_embeddings(input_ids).shape, (self.model_tester.batch_size, config.hidden_size)
        )
        self.assertEqual(model.get_multivector_embeddings(input_ids).shape[:2], input_ids.shape)
        for width in (-4, 0, config.hidden_size + 1):
            with self.subTest(dense_dim=width), self.assertRaises(ValueError):
                model(input_ids=input_ids, dense_dim=width)

    def test_retrieval_heads_can_be_selected_individually(self):
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = NeoMMEForRetrieval(config).to(torch_device).eval()

        self.assertIsNone(model(input_ids=input_ids, output_dense=False).dense_embeddings)
        self.assertIsNone(model(input_ids=input_ids, output_multivector=False).multivector_embeddings)
        with self.assertRaises(ValueError):
            model(input_ids=input_ids, output_dense=False, output_multivector=False)

    def test_fully_padded_row_pools_to_a_finite_vector(self):
        """A row with no real token must not turn into NaN through the pooler's softmax."""
        config, input_ids, input_mask, _ = self.model_tester.prepare_config_and_inputs()
        input_mask[0] = 0
        model = NeoMMEForRetrieval(config).to(torch_device).eval()
        output = model(input_ids=input_ids, attention_mask=input_mask)

        self.assertTrue(torch.isfinite(output.dense_embeddings).all())
        self.assertTrue(torch.isfinite(output.multivector_embeddings).all())
        self.assertTrue((output.multivector_embeddings[0] == 0).all())


# NOTE: `Hcompany/neomme-250M-retrieval-dev-transformers-v0.3` is a PRIVATE staging repo, so these tests
# need Hub credentials with access to it. Repoint them at the public checkpoint at release.
@slow
@require_torch
@require_vision
class NeoMMEModelIntegrationTest(unittest.TestCase):
    model_name: ClassVar[str] = "Hcompany/neomme-250M-retrieval-dev-transformers-v0.3"
    # Parity is only ever gated in float32; bf16 drift is documented separately and never asserted on.
    model_dtype: ClassVar["torch.dtype"] = torch.float32 if is_torch_available() else None

    def setUp(self):
        self.processor = NeoMMEProcessor.from_pretrained(self.model_name)
        self.model = (
            NeoMMEForRetrieval.from_pretrained(self.model_name, dtype=self.model_dtype).to(torch_device).eval()
        )

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_integration_test(self):
        """The model retrieves the right page for every query of a small, easy dataset."""
        dataset = load_dataset("hf-internal-testing/document-visual-retrieval-test", split="test")

        batch_images = self.processor(images=dataset["image"][:]).to(torch_device)
        batch_queries = self.processor(text=dataset["query"][:], text_role="query").to(torch_device)

        with torch.inference_mode():
            image_embeddings = self.model(**batch_images).multivector_embeddings
            query_embeddings = self.model(**batch_queries).multivector_embeddings

        scores = self.processor.score_retrieval(query_embeddings, image_embeddings)

        self.assertEqual(scores.ndim, 2)
        self.assertEqual(scores.shape, (len(dataset), len(dataset)))
        # Every query's best match is its own page, i.e. the argmax sits on the diagonal.
        self.assertTrue((scores.argmax(dim=1) == torch.arange(len(dataset), device=scores.device)).all())
        # MaxSim over L2-normalized tokens, averaged over query tokens, is bounded by [-1, 1].
        self.assertTrue(((scores >= -1.0) & (scores <= 1.0)).all())

        # Measured against the research implementation on the same weights, which agreed to 4.9e-07.
        expected_scores = torch.tensor(
            [
                [0.8281, 0.6679, 0.8145],
                [0.7385, 0.8536, 0.7621],
                [0.7486, 0.7014, 0.8988],
            ],
            dtype=scores.dtype,
        )
        torch.testing.assert_close(scores, expected_scores, rtol=1e-3, atol=1e-3)

    def test_dense_head_integration_test(self):
        """The dense (latent-attention pooled) head retrieves the right page too, scored by cosine."""
        dataset = load_dataset("hf-internal-testing/document-visual-retrieval-test", split="test")

        batch_images = self.processor(images=dataset["image"][:]).to(torch_device)
        batch_queries = self.processor(text=dataset["query"][:], text_role="query").to(torch_device)

        with torch.inference_mode():
            image_embeddings = self.model(**batch_images, output_multivector=False).dense_embeddings
            query_embeddings = self.model(**batch_queries, output_multivector=False).dense_embeddings

        self.assertEqual(image_embeddings.shape, (len(dataset), self.model.config.hidden_size))
        torch.testing.assert_close(
            image_embeddings.norm(dim=-1), torch.ones_like(image_embeddings[:, 0]), rtol=1e-3, atol=1e-3
        )

        scores = self.processor.score_retrieval(query_embeddings, image_embeddings)
        self.assertEqual(scores.shape, (len(dataset), len(dataset)))
        self.assertTrue((scores.argmax(dim=1) == torch.arange(len(dataset), device=scores.device)).all())

        expected_scores = torch.tensor(
            [
                [0.6396, 0.4721, 0.5920],
                [0.6066, 0.6850, 0.6375],
                [0.5440, 0.5334, 0.6988],
            ],
            dtype=scores.dtype,
        )
        torch.testing.assert_close(scores, expected_scores, rtol=1e-3, atol=1e-3)

    def test_text_document_retrieval(self):
        """Text passages go through the `<doc>` side, so a query beats a distractor on the text path too."""
        queries = ["How many people live in the capital of France?", "What colour is a ripe banana?"]
        documents = [
            "Paris is the capital of France and has a population of about 2.1 million people.",
            "Bananas start out green and turn yellow as they ripen.",
        ]

        batch_queries = self.processor(text=queries, text_role="query").to(torch_device)
        batch_documents = self.processor(text=documents, text_role="document").to(torch_device)

        with torch.inference_mode():
            query_embeddings = self.model(**batch_queries).multivector_embeddings
            document_embeddings = self.model(**batch_documents).multivector_embeddings

        scores = self.processor.score_retrieval(query_embeddings, document_embeddings)
        self.assertTrue((scores.argmax(dim=1) == torch.arange(len(documents), device=scores.device)).all())

        expected_scores = torch.tensor([[0.8280, 0.3310], [0.3668, 0.6765]], dtype=scores.dtype)
        torch.testing.assert_close(scores, expected_scores, rtol=1e-3, atol=1e-3)

    def test_masked_lm_logits(self):
        """The tied factorized decode, on real weights.

        Loaded from the retrieval checkpoint: `NeoMMEForMaskedLM` adds no parameters, so it takes that
        repo's backbone through `base_model_prefix`. These are the phase-2 trunk's weights rather than the
        pretrained one's, which is fine here — the test pins the decode's arithmetic, not reading quality.
        """
        model = NeoMMEForMaskedLM.from_pretrained(self.model_name, dtype=self.model_dtype).to(torch_device).eval()
        inputs = self.processor(text=["The capital of France is <mask>."], text_role="document").to(torch_device)

        with torch.inference_mode():
            logits = model(**inputs).logits

        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[-1], model.config.vocab_size)
        self.assertTrue(torch.isfinite(logits).all())

        expected_slice = torch.tensor(
            [[1.0907, 0.1332, 2.8313], [21.4681, 17.0028, 18.4514], [-0.8730, -2.0681, -5.4430]],
            dtype=logits.dtype,
        )
        torch.testing.assert_close(logits[0, :3, :3], expected_slice, rtol=1e-3, atol=1e-3)

        # The decode is also semantically wired, not merely numerically stable: filling the mask should
        # reach for France.
        mask_id = self.processor.tokenizer.mask_token_id
        mask_position = (inputs["input_ids"][0] == mask_id).nonzero().flatten()[0]
        top_tokens = self.processor.tokenizer.convert_ids_to_tokens(logits[0, mask_position].topk(5).indices)
        self.assertTrue(any("Fran" in token for token in top_tokens), f"expected France in {top_tokens}")

    def test_both_heads_come_from_one_backbone_pass(self):
        """Asking for both heads at once matches asking for each on its own."""
        dataset = load_dataset("hf-internal-testing/document-visual-retrieval-test", split="test")
        batch_images = self.processor(images=dataset["image"][:1]).to(torch_device)

        with torch.inference_mode():
            both = self.model(**batch_images)
            multivector_only = self.model(**batch_images, output_dense=False)
            dense_only = self.model(**batch_images, output_multivector=False)

        torch.testing.assert_close(both.multivector_embeddings, multivector_only.multivector_embeddings)
        torch.testing.assert_close(both.dense_embeddings, dense_only.dense_embeddings)
