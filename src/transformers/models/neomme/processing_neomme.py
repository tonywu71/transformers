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
"""Processor class for NeoMME."""

from typing import Any, Literal

import numpy as np
import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


def _pad_grids(embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Variable-length `(length, dim)` token grids -> padded `(batch, max_length, dim)` plus a bool mask."""
    lengths = torch.tensor([grid.shape[0] for grid in embeddings])
    mask = torch.arange(int(lengths.max()))[None, :] < lengths[:, None]
    padded = torch.zeros(*mask.shape, embeddings[0].shape[-1], dtype=embeddings[0].dtype)
    for index, grid in enumerate(embeddings):
        padded[index, : grid.shape[0]] = grid
    return padded, mask


def _as_padded_grids(embeddings: torch.Tensor | list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept either a padded 3-D tensor or a list of `(length, dim)` grids; return `(grids, mask)`.

    A 3-D tensor is assumed to come straight from [`NeoMMEForRetrieval`], whose multi-vector head zeroes
    padding rows exactly, so all-zero rows are the padding mask.
    """
    if isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3:
        return embeddings, embeddings.abs().sum(-1) > 0
    return _pad_grids(list(embeddings))


def maxsim_scores(
    query_grids: torch.Tensor,
    passage_grids: torch.Tensor,
    query_mask: torch.Tensor,
    passage_mask: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """ColBERT/ColPali late interaction: sum over query tokens of the max cosine over passage tokens.

    Padded passage tokens are dropped from the max and padded query tokens contribute nothing.
    `normalize` divides by the query length, making the score a MEAN over query tokens so one
    temperature stays valid across very different query lengths — this is what NeoMME was trained and
    evaluated with. A fully padded passage is floored to a finite never-match `-1.0`.
    """
    query_grids = torch.nn.functional.normalize(query_grids.float(), dim=-1) * query_mask[..., None]
    passage_grids = torch.nn.functional.normalize(passage_grids.float(), dim=-1)
    similarity = torch.einsum("qid,pjd->qpij", query_grids, passage_grids)
    similarity = similarity.masked_fill(~passage_mask[None, :, None, :], torch.finfo(similarity.dtype).min)
    scores = similarity.max(dim=-1).values.sum(dim=-1)

    if normalize:
        scores = scores / query_mask.sum(-1, keepdim=True).clamp_min(1).to(scores.dtype)
    return scores.masked_fill(~passage_mask.any(dim=-1)[None, :], -1.0)


class NeoMMEProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": "longest"},
        "images_kwargs": {"do_convert_rgb": True},
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class NeoMMEProcessor(ProcessorMixin):
    r"""
    Constructs a NeoMME processor: a tokenizer plus a [`NeoMMEImageProcessor`], with the query/document
    marker convention baked in.

    Queries and documents are opposite retrieval sides encoded in separate forward passes, so exactly one
    modality is accepted per call:

    - `process_queries` emits `[<query>] + tokens + query_expand * [<mask>]` (ColBERT query augmentation;
      the `<mask>` token is also the model's masked-diffusion fill token, so its representation is the
      strongest one available).
    - `process_text_documents` emits `[<doc>] + tokens`.
    - `process_images` emits `<doc> <img>` followed by the row-major patch grid with a `<row>` break token
      at the end of every patch row, plus the two-axis `position_ids` the grid needs.

    """

    valid_processor_kwargs = NeoMMEProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        query_token: str = "<query>",
        document_token: str = "<doc>",
        image_token: str = "<img>",
        row_token: str = "<row>",
        query_expand: int = 10,
        **kwargs,
    ):
        r"""
        query_token (`str`, *optional*, defaults to `"<query>"`):
            Marker token prefixed to every query.
        document_token (`str`, *optional*, defaults to `"<doc>"`):
            Marker token prefixed to every document, text or image.
        image_token (`str`, *optional*, defaults to `"<img>"`):
            Placeholder token the patch embeddings are scattered into.
        row_token (`str`, *optional*, defaults to `"<row>"`):
            Token that closes every row of the image patch grid.
        query_expand (`int`, *optional*, defaults to 10):
            Number of `<mask>` buffer tokens appended to every query.
        """
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.query_token = query_token
        self.document_token = document_token
        self.image_token = image_token
        self.row_token = row_token
        self.query_expand = query_expand

    @property
    def query_augmentation_token(self) -> str:
        """The token appended to queries as learned query expansion (the model's `<mask>` token)."""
        return self.tokenizer.mask_token

    @property
    def model_input_names(self) -> list[str]:
        return ["input_ids", "attention_mask", "position_ids", "pixel_values", "image_grid_hw"]

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_role: Literal["query", "document"] = "query",
        **kwargs: Unpack[NeoMMEProcessorKwargs],
    ) -> BatchFeature:
        r"""
        text_role (`str`, *optional*, defaults to `"query"`):
            Which marker convention `text` gets: `"query"` prefixes `<query>` and appends the `<mask>`
            expansion, `"document"` prefixes `<doc>`. Ignored when `images` is passed — images are always
            documents.

        Returns:
            [`BatchFeature`] with `input_ids` and `attention_mask`, plus `position_ids`, `pixel_values` and
            `image_grid_hw` on the image path.
        """
        if (text is None) == (images is None):
            raise ValueError(
                "Pass exactly one of `text` or `images`: they are opposite retrieval sides, encoded in "
                "separate forward passes."
            )

        output_kwargs = self._merge_kwargs(
            NeoMMEProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        if images is not None:
            return self.process_images(images, **output_kwargs["images_kwargs"])

        if isinstance(text, str):
            text = [text]
        # What the caller actually named, flat or nested, as opposed to what `_merge_kwargs` injected.
        requested = set(kwargs) | set(kwargs.get("text_kwargs", {}))
        text_kwargs = self._supported_text_kwargs(output_kwargs["text_kwargs"], requested)
        if text_role == "query":
            return self.process_queries(text, **text_kwargs)
        if text_role == "document":
            return self.process_text_documents(text, **text_kwargs)
        raise ValueError(f"text_role={text_role!r} is not supported: expected 'query' or 'document'.")

    def process_queries(
        self,
        text: list[str] | str,
        max_length: int | None = None,
        padding: bool | str = "longest",
        return_tensors: str | None = "pt",
    ) -> BatchFeature:
        """Tokenize queries, prefix `<query>`, append the `<mask>` expansion, then pad the batch.

        When `max_length` is set the CONTENT is truncated to `max_length - 1 - query_expand` so the marker
        and the full expansion suffix always survive.
        """
        if isinstance(text, str):
            text = [text]
        marker_ids = self._marker_ids()
        expansion = [marker_ids["mask"]] * self.query_expand

        content_limit = None
        if max_length is not None:
            content_limit = max_length - 1 - self.query_expand
            if content_limit < 0:
                raise ValueError(
                    f"query_expand={self.query_expand} leaves no room for content inside max_length={max_length}"
                )

        encodings = self.tokenizer(list(text), add_special_tokens=False)["input_ids"]
        sequences = [[marker_ids["query"]] + list(ids)[:content_limit] + expansion for ids in encodings]
        return self._pad_sequences(sequences, padding=padding, max_length=max_length, return_tensors=return_tensors)

    def process_text_documents(
        self,
        text: list[str] | str,
        max_length: int | None = None,
        padding: bool | str = "longest",
        return_tensors: str | None = "pt",
    ) -> BatchFeature:
        """Tokenize text passages, prefix `<doc>` (no query expansion), then pad the batch."""
        if isinstance(text, str):
            text = [text]
        marker_ids = self._marker_ids()
        content_limit = None if max_length is None else max_length - 1

        # An empty string tokenizes to nothing, which would leave a marker-only document.
        encodings = self.tokenizer([passage or " " for passage in text], add_special_tokens=False)["input_ids"]
        sequences = [[marker_ids["document"]] + list(ids)[:content_limit] for ids in encodings]
        return self._pad_sequences(sequences, padding=padding, max_length=max_length, return_tensors=return_tensors)

    def process_images(self, images: ImageInput, **kwargs) -> BatchFeature:
        """Patchify each page and lay it out as `<doc> <img>` + the row-major patch grid.

        Every image restarts its position grid at `(0, 0)`: the two marker tokens take the diagonal
        positions `(0, 0)` and `(1, 1)`, and patch `(row, column)` takes position `(2 + row, 2 + column)`.
        Carrying a running offset across a batch corrupts every image after the first.
        """
        if is_valid_image(images):
            images = [images]
        return_tensors = kwargs.setdefault("return_tensors", "pt")
        image_inputs = self.image_processor(images=images, **kwargs)
        marker_ids = self._marker_ids()

        sequences: list[list[int]] = []
        positions: list[np.ndarray] = []
        for grid_height, grid_width in image_inputs["image_grid_hw"].tolist():
            ids, position_ids = self._encode_image_grid(grid_height, grid_width, marker_ids)
            sequences.append(ids)
            positions.append(position_ids)

        batch = self._pad_sequences(sequences, positions, return_tensors=return_tensors)
        batch["pixel_values"] = image_inputs["pixel_values"]
        batch["image_grid_hw"] = image_inputs["image_grid_hw"]
        return batch

    def score_retrieval(
        self,
        query_embeddings: torch.Tensor | list[torch.Tensor],
        passage_embeddings: torch.Tensor | list[torch.Tensor],
        normalize: bool = True,
        output_dtype: torch.dtype | None = None,
        output_device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """`(num_queries, num_passages)` scores: MaxSim for multi-vector inputs, cosine for dense ones.

        Args:
            query_embeddings / passage_embeddings:
                Either 3-D padded multi-vector grids / lists of `(length, dim)` grids, or 2-D dense
                matrices. Both sides must be of the same kind.
            normalize (`bool`, *optional*, defaults to `True`):
                MaxSim only: divide by the query length so the score is a mean over query tokens.
        """
        if len(query_embeddings) == 0 or len(passage_embeddings) == 0:
            raise ValueError("Both `query_embeddings` and `passage_embeddings` must be non-empty")

        if self._is_multi_vector(passage_embeddings):
            query_grids, query_mask = _as_padded_grids(query_embeddings)
            passage_grids, passage_mask = _as_padded_grids(passage_embeddings)
            scores = maxsim_scores(query_grids, passage_grids, query_mask, passage_mask, normalize=normalize)
        else:
            queries = self._as_dense(query_embeddings)
            passages = self._as_dense(passage_embeddings)
            scores = (
                torch.nn.functional.normalize(queries.float(), dim=-1)
                @ torch.nn.functional.normalize(passages.float(), dim=-1).t()
            )

        return scores.to(output_dtype or scores.dtype).to(output_device)

    def _marker_ids(self) -> dict[str, int]:
        """Marker name -> token id, refusing a tokenizer that does not carry the markers.

        `convert_tokens_to_ids` answers `unk_token_id` for a token it has never seen, so checking for
        `None` catches almost nothing: the markers would silently degrade into `<unk>`, which costs recall
        without failing anything.
        """
        ids = {
            "query": self.tokenizer.convert_tokens_to_ids(self.query_token),
            "document": self.tokenizer.convert_tokens_to_ids(self.document_token),
            "image": self.tokenizer.convert_tokens_to_ids(self.image_token),
            "row": self.tokenizer.convert_tokens_to_ids(self.row_token),
            "mask": self.tokenizer.mask_token_id,
        }
        unknown_id = self.tokenizer.unk_token_id
        missing = [name for name, token_id in ids.items() if token_id is None or token_id == unknown_id]
        if missing:
            raise ValueError(f"The tokenizer is missing NeoMME marker tokens: {missing}")
        return ids

    def _encode_image_grid(
        self, grid_height: int, grid_width: int, marker_ids: dict[str, int]
    ) -> tuple[list[int], np.ndarray]:
        """`<doc> <img>` + `grid_height` rows of `grid_width` patch tokens each closed by a `<row>` break."""
        grid = np.full((grid_height, grid_width + 1), marker_ids["image"], dtype=np.int64)
        grid[:, grid_width] = marker_ids["row"]
        ids = [marker_ids["document"], marker_ids["image"], *grid.ravel().tolist()]

        positions = np.empty((len(ids), 2), dtype=np.int64)
        positions[0] = (0, 0)
        positions[1] = (1, 1)
        rows = np.broadcast_to(np.arange(grid_height)[:, None], grid.shape)
        columns = np.broadcast_to(np.arange(grid_width + 1)[None, :], grid.shape)
        positions[2:, 0] = 2 + rows.ravel()
        positions[2:, 1] = 2 + columns.ravel()
        return ids, positions

    def _pad_sequences(
        self,
        sequences: list[list[int]],
        positions: list[np.ndarray] | None = None,
        padding: bool | str = "longest",
        max_length: int | None = None,
        return_tensors: str | None = "pt",
    ) -> BatchFeature:
        """Right-pad `sequences` (and their two-axis positions) into `(batch, length)` tensors."""
        length = self._padded_length([len(ids) for ids in sequences], padding, max_length)
        pad_token_id = self.tokenizer.pad_token_id or 0

        data: dict[str, Any] = {
            "input_ids": [ids + [pad_token_id] * (length - len(ids)) for ids in sequences],
            "attention_mask": [[1] * len(ids) + [0] * (length - len(ids)) for ids in sequences],
        }
        if positions is not None:
            # (2, batch, length): index 0 is the M-RoPE row axis, index 1 the column axis.
            grid = np.zeros((2, len(sequences), length), dtype=np.int64)
            for index, image_positions in enumerate(positions):
                grid[:, index, : image_positions.shape[0]] = image_positions.T
            data["position_ids"] = grid.tolist()
        return BatchFeature(data=data, tensor_type=return_tensors)

    def _padded_length(self, lengths: list[int], padding: bool | str, max_length: int | None) -> int:
        """The width every row is padded to, following the tokenizer's `padding` vocabulary."""
        longest = max(lengths)
        if padding in ("max_length",):
            if max_length is None:
                raise ValueError("padding='max_length' needs a `max_length`.")
            if max_length < longest:
                raise ValueError(
                    f"max_length={max_length} is shorter than the longest encoded row ({longest}); padding to it "
                    "would drop tokens."
                )
            return max_length
        if padding in (False, "do_not_pad") and longest != min(lengths):
            raise ValueError(
                "padding=False cannot return a single tensor for rows of different lengths. Pass "
                "padding='longest', or encode one sequence at a time."
            )
        return longest

    def _supported_text_kwargs(self, text_kwargs: dict[str, Any], requested: set[str]) -> dict[str, Any]:
        """Keep the text kwargs this processor implements, and refuse the rest rather than drop it.

        The marker convention is not negotiable: `add_special_tokens` or `stride` would silently corrupt the
        layout the model was trained on, and a kwarg that is quietly ignored is worse than one that raises.
        Truncation needs no flag — a `max_length` always truncates the content and never the markers.

        Only `requested` — what the CALLER passed — is refusable. `_merge_kwargs` also folds in defaults from
        `tokenizer.init_kwargs`, so a `tokenizer_config.json` carrying `padding_side` (as every tokenizer
        copied from Llama, Qwen or Mistral does) would otherwise make every text call raise over a kwarg the
        caller never wrote. An injected value is dropped instead: this processor always right-pads. An
        explicit `padding_side=` from the caller still raises rather than being silently ignored.
        """
        supported = {
            name: text_kwargs[name] for name in ("max_length", "padding", "return_tensors") if name in text_kwargs
        }
        if text_kwargs.get("truncation") is False and text_kwargs.get("max_length") is not None:
            raise ValueError(
                "truncation=False with a max_length is not supported: the marker and query-expansion layout "
                "is fixed, so content past max_length is always dropped."
            )
        unsupported = sorted((set(text_kwargs) & requested) - set(supported) - {"truncation"})
        if unsupported:
            raise ValueError(f"NeoMMEProcessor does not implement these text kwargs: {unsupported}.")
        return supported

    def _is_multi_vector(self, embeddings: torch.Tensor | list[torch.Tensor]) -> bool:
        if isinstance(embeddings, torch.Tensor):
            return embeddings.dim() == 3
        return embeddings[0].dim() == 2

    def _as_dense(self, embeddings: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        return embeddings if isinstance(embeddings, torch.Tensor) else torch.stack(list(embeddings))


__all__ = ["NeoMMEProcessor"]
