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
"""NeoMME model configuration."""

from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring
@strict
class NeoMMEConfig(PreTrainedConfig):
    r"""
    embedding_rank (`int`, *optional*, defaults to 256):
        Inner dimension of the factorized INPUT embedding table (ALBERT-style): the word table is
        `vocab_size x embedding_rank`, projected up to `hidden_size`, which keeps a 128k-token vocabulary
        cheap. It is also the width the tied masked-LM decode projects back down to, so that head costs
        no parameters of its own. Sizes the input side only — the retrieval output width is
        `embedding_dim`.
    global_attn_every_n_layers (`int`, *optional*, defaults to 6):
        Stride of the global layers: layer `i` runs full bidirectional attention when
        `(i + 1) % global_attn_every_n_layers == 0`, and the **last layer is always global**. Every other
        layer runs bidirectional sliding-window attention. This only generates `layer_types`; when both
        are given they must agree, and `None` declares a hand-written pattern that no stride produces.
    sliding_window_short (`int`, *optional*, defaults to 256):
        Sliding-window **half-width per side** (the band is `abs(i - j) <= sliding_window_short`) of the
        short sliding-window layers.
    sliding_window_long (`int`, *optional*, defaults to 1024):
        Sliding-window half-width of the long sliding-window layers. Sliding layers alternate short/long
        by sliding-layer ordinal parity (the second, fourth, ... sliding layer is long), counted over
        sliding layers only so the pattern is independent of where the global layers fall. Set it equal to
        `sliding_window_short` for a single band width.
    use_value_embeds (`bool`, *optional*, defaults to `True`):
        Whether to add a per-token value-embedding table into the value stream of the first and last
        global layers. The 250M release carries the table; a variant trained without it drops 13% of the
        parameters, which is why this stays configurable while the rest of the block does not.
    patch_size (`int`, *optional*, defaults to 32):
        Side, in pixels, of one image patch token. NeoMME is vision-tower-free: a flattened
        `3 * patch_size ** 2` RGB block is fed straight to the patch stem, so the image token count is
        set entirely by the resolution the image processor emits.
    embedding_dim (`int`, *optional*, defaults to 128):
        Width of the per-token multi-vector embeddings [`NeoMMEForRetrieval`] OUTPUTS, i.e. of its
        late-interaction projection (the ColBERT/Col* convention). Nothing to do with `embedding_rank`,
        which sizes the input embedding table.
    document_token_id (`int`, *optional*, defaults to 5):
        Id of the `<doc>` marker token that opens every document. Needed to tell the single `<img>` marker
        that follows it apart from the `<img>` patch placeholders, which share its id.
    image_token_id (`int`, *optional*, defaults to 6):
        Id of the `<img>` token. It serves double duty: one marker directly after `<doc>` announces that
        the document is an image, and every following occurrence is a patch placeholder that a patch
        embedding is scattered into.

    Only the ids the FORWARD needs are config fields. The rest of the marker block — `<query>`, `<row>`,
    `<mask>`, `<eos>` — is laid out by [`NeoMMEProcessor`], which resolves it through the tokenizer, so
    carrying those ids here would duplicate tokenizer state that nothing reads back.

    Examples:

    ```python
    >>> from transformers import NeoMMEModel, NeoMMEConfig

    >>> # Initializing a NeoMME 256M style configuration
    >>> configuration = NeoMMEConfig()

    >>> # Initializing a model from that configuration
    >>> model = NeoMMEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "neomme"
    # Gemma-4-style local/global split: the sliding layers run a short-wavelength spectrum that fits
    # inside their window, the global layers a long one.
    default_theta = {"full_attention": 1_000_000.0, "sliding_attention": 10_000.0}
    # Global layers keep most head dims position-blind (NoPE) for content matching; sliding layers spend
    # every dim on fine 2-axis position.
    default_partial_rotary_factor = {"full_attention": 0.25, "sliding_attention": 1.0}
    ignore_keys_at_rope_validation = {"partial_rotary_factor"}
    # Research switches that are FIXED here: every value NeoMME was released with is the one the port
    # implements, so the alternatives are unreachable rather than merely untested. They are still refused
    # by name, because a research `config.json` carrying one would otherwise be read as if it agreed.
    # `depth_scale` is the dangerous one: it changes no tensor shape, so a checkpoint trained without it
    # would load clean and be quietly wrong.
    fixed_architecture = {"use_xsa": True, "depth_scale": True, "patch_stem": "mlp", "cheap_mixer": "swa"}

    vocab_size: int = 131072
    embedding_rank: int = 256
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 17
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    max_position_embeddings: int = 16384
    norm_eps: float = 1e-6
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0

    layer_types: list[str] | None = None
    global_attn_every_n_layers: int | None = 6
    rope_parameters: dict[Literal["full_attention", "sliding_attention"], dict] | None = None
    sliding_window_short: int = 256
    sliding_window_long: int = 1024

    use_value_embeds: bool = True

    patch_size: int = 32
    embedding_dim: int = 128

    pad_token_id: int | None = 0
    document_token_id: int | None = 5
    image_token_id: int | None = 6
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self._refuse_fixed_architecture(kwargs)
        if not 0 < self.sliding_window_short <= self.sliding_window_long:
            raise ValueError(
                f"expected 0 < sliding_window_short <= sliding_window_long, got {self.sliding_window_short} "
                f"and {self.sliding_window_long}. Pass two equal widths for a single band; the research "
                "encoding of `sliding_window_long = 0` for 'uniform' is resolved by the conversion script."
            )
        if self.layer_types is None:
            if self.global_attn_every_n_layers is None:
                raise ValueError(
                    "Pass either `layer_types` or `global_attn_every_n_layers`: with both unset there is no "
                    "attention pattern to build. `global_attn_every_n_layers=None` only means 'this pattern came "
                    "from no stride', which requires an explicit `layer_types`."
                )
            self.layer_types = self.generate_layer_types(self.global_attn_every_n_layers)
        else:
            self._validate_layer_types()

        super().__post_init__(**kwargs)

    def generate_layer_types(self, global_attn_every_n_layers: int) -> list[str]:
        """The attention pattern for a stride: every Nth layer is global, and so is the LAST one."""
        return [
            "full_attention"
            if (i + 1) % global_attn_every_n_layers == 0 or i == self.num_hidden_layers - 1
            else "sliding_attention"
            for i in range(self.num_hidden_layers)
        ]

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        # A flat `rope_theta` is the standard HF knob, so it overrides both per-layer-type defaults. Popping
        # it keeps it out of `config.json`, where it would otherwise sit next to `rope_parameters` reading
        # as authoritative while nothing ever consumed it.
        rope_theta = kwargs.pop("rope_theta", None)
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Only the layer types this model actually has: a homogeneous `layer_types` whose rope keys are a
        # strict superset sends `standardize_rope_params` down its single-global-dict branch, which writes
        # flat keys into `rope_parameters` and fails the annotation.
        for layer_type in set(self.layer_types):
            layer_params = self.rope_parameters.setdefault(layer_type, {})
            if rope_scaling is not None:
                layer_params.update(rope_scaling)
            layer_params.setdefault("rope_type", "default")
            layer_params.setdefault("rope_theta", rope_theta or self.default_theta[layer_type])
            layer_params.setdefault("partial_rotary_factor", self.default_partial_rotary_factor[layer_type])

        self.standardize_rope_params()
        self._validate_rotary_dims()
        return kwargs

    def _refuse_fixed_architecture(self, kwargs: dict) -> None:
        """Reject a research kwarg that names an architecture this port does not implement.

        These arrive from a research `config.json`, where they were ablation switches — `use_xsa=False`,
        `patch_stem="linear"` and `cheap_mixer="gdn"` are all research DEFAULTS that no released NeoMME
        uses. Accepting and ignoring them would describe a model the weights do not match.
        """
        for name, supported in self.fixed_architecture.items():
            if name in kwargs and kwargs[name] != supported:
                raise ValueError(
                    f"{name}={kwargs[name]!r} is not supported: the transformers port implements "
                    f"{name}={supported!r} only, which is what every released NeoMME was trained with. "
                    "Convert such a checkpoint in the research repo instead."
                )
            kwargs.pop(name, None)

    def _validate_layer_types(self) -> None:
        """Check an explicitly-given `layer_types`, and that it agrees with any given stride.

        `layer_types` is what gets serialized, so on reload it is the source of truth and
        `global_attn_every_n_layers` is only the stride it was generated from. Letting the two disagree
        silently would mean a config that reads one way and behaves another, so a conflict raises — pass
        `global_attn_every_n_layers=None` to declare a pattern that no single stride produces.
        """
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but num_hidden_layers is "
                f"{self.num_hidden_layers}; there must be exactly one entry per layer."
            )
        unknown = sorted(set(self.layer_types) - {"full_attention", "sliding_attention"})
        if unknown:
            raise ValueError(f"layer_types contains unknown values {unknown}; expected full/sliding_attention.")

        if self.global_attn_every_n_layers is None:
            return
        expected = self.generate_layer_types(self.global_attn_every_n_layers)
        if self.layer_types != expected:
            raise ValueError(
                f"layer_types disagrees with global_attn_every_n_layers={self.global_attn_every_n_layers}, "
                f"which generates {expected} but got {self.layer_types}. Fix the stride, or pass "
                "global_attn_every_n_layers=None to keep a hand-written pattern."
            )

    def _validate_rotary_dims(self) -> None:
        """Check every layer type rotates a multiple of 4 dims.

        The two M-RoPE axes take alternating frequency pairs, so the rotating dims divide by 2 twice.
        A factor that breaks that used to be rounded down inside the model, which produced a working
        model with a narrower spectrum than the config advertised — unrecoverable once weights were
        trained against it, and invisible in `config.json`.
        """
        for layer_type in sorted(set(self.layer_types)):
            partial_rotary_factor = self.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
            rotary_dim = int(self.head_dim * partial_rotary_factor)
            if rotary_dim % 4:
                raise ValueError(
                    f"rope_parameters[{layer_type!r}]['partial_rotary_factor']={partial_rotary_factor} rotates "
                    f"{rotary_dim} of head_dim={self.head_dim} dims, which is not a multiple of 4: the two "
                    f"M-RoPE axes consume frequencies in alternating pairs. Nearest usable factors: "
                    f"{(rotary_dim - rotary_dim % 4) / self.head_dim} or "
                    f"{(rotary_dim + 4 - rotary_dim % 4) / self.head_dim}."
                )

    @property
    def patch_dim(self) -> int:
        """Width of one flattened RGB patch, `3 * patch_size ** 2`."""
        return 3 * self.patch_size**2

    @property
    def layer_window_sizes(self) -> list[int | None]:
        """Per-layer sliding-window half-width; `None` on global layers.

        Sliding layers alternate short/long by their ordinal among sliding layers, so the pattern does
        not shift when the global layers move.
        """
        windows: list[int | None] = []
        sliding_idx = 0
        for layer_type in self.layer_types:
            if layer_type == "full_attention":
                windows.append(None)
                continue
            windows.append(self.sliding_window_long if sliding_idx % 2 else self.sliding_window_short)
            sliding_idx += 1
        return windows


__all__ = ["NeoMMEConfig"]
