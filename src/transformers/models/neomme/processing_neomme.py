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
    lengths = torch.tensor([grid.shape[0] for grid in embeddings])  # (batch_size,)
    mask = torch.arange(int(lengths.max()))[None, :] < lengths[:, None]  # (batch_size, max_length)
    padded = torch.zeros(*mask.shape, embeddings[0].shape[-1], dtype=embeddings[0].dtype)  # (batch_size, max_length, dim)
    for index, grid in enumerate(embeddings):
        padded[index, : grid.shape[0]] = grid
    return padded, mask


def _as_padded_grids(embeddings: torch.Tensor | list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept either a padded 3-D tensor or a list of token grids."""
    if isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3:
        return embeddings, embeddings.abs().sum(-1) > 0  # mask: (batch_size, max_length)
    return _pad_grids(list(embeddings))


def maxsim_scores(
    query_grids: torch.Tensor,
    passage_grids: torch.Tensor,
    query_mask: torch.Tensor,
    passage_mask: torch.Tensor,
    normalize: bool = True,  # TODO: remove if we don't ship MeanMaxSim
) -> torch.Tensor:
    """ColBERT-style late interaction (MaxSim) over multi-vector embeddings.

    For each query token, take the maximum cosine similarity over passage tokens, then sum those
    maxima. Embeddings are always L2-normalized along the last dimension before scoring; padding
    rows are ignored via the masks. Empty passages (no valid tokens) score `-1`.

    Args:
        query_grids (`torch.Tensor` of shape `(num_queries, query_length, dim)`):
            Query token embeddings. Padding rows should be zero (or otherwise excluded by
            `query_mask`).
        passage_grids (`torch.Tensor` of shape `(num_passages, passage_length, dim)`):
            Passage token embeddings. Same padding convention as `query_grids`.
        query_mask (`torch.Tensor` of shape `(num_queries, query_length)`):
            Bool mask that is `True` for real query tokens and `False` for padding.
        passage_mask (`torch.Tensor` of shape `(num_passages, passage_length)`):
            Bool mask that is `True` for real passage tokens and `False` for padding.
        normalize (`bool`, *optional*, defaults to `True`):
            If `True`, divide each score by the number of non-padding query tokens so scores stay
            roughly in `[-1, 1]` regardless of query length. If `False`, return the raw ColBERT
            sum (scales with query length). This is **not** about L2-normalizing the vectors —
            that always happens.

    Returns:
        `torch.Tensor` of shape `(num_queries, num_passages)`.
    """
    query_grids = (
        torch.nn.functional.normalize(query_grids.float(), dim=-1) * query_mask[..., None]
    )  # (num_queries, query_length, dim)
    passage_grids = torch.nn.functional.normalize(passage_grids.float(), dim=-1)  # (num_passages, passage_length, dim)
    similarity = torch.einsum(
        "qid,pjd->qpij", query_grids, passage_grids
    )  # (num_queries, num_passages, query_length, passage_length)
    similarity = similarity.masked_fill(~passage_mask[None, :, None, :], torch.finfo(similarity.dtype).min)
    scores = similarity.max(dim=-1).values.sum(dim=-1)  # (num_queries, num_passages)

    if normalize:  # TODO: remove if we don't ship MeanMaxSim
        scores = scores / query_mask.sum(-1, keepdim=True).clamp_min(1).to(scores.dtype)
    return scores.masked_fill(~passage_mask.any(dim=-1)[None, :], -1.0)


class NeoMMEProcessorKwargs(ProcessingKwargs, total=False):
    # Every call default lives where it applies instead: `padding` / `return_tensors` on the
    # `process_*` signatures, `do_convert_rgb` on the image processors. The empty dict still has to be
    # here, since `_merge_kwargs` reads `_defaults` unguarded and TypedDict subclasses do not inherit it.
    _defaults = {}


@auto_docstring
class NeoMMEProcessor(ProcessorMixin):
    r"""
    Constructs a NeoMME processor that wraps a tokenizer and image processor.

    Queries and documents are encoded in separate forward passes, so exactly one of `text` or `images`
    is accepted per call. Queries are prefixed with `<query>` and expanded with `<mask>` tokens;
    documents are prefixed with `<doc>`; image documents also include a patch grid and two-axis
    `position_ids`.
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
        """Tokenize queries with `<query>` prefix and `<mask>` expansion."""
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
        """Tokenize text documents with `<doc>` prefix."""
        if isinstance(text, str):
            text = [text]
        marker_ids = self._marker_ids()
        content_limit = None if max_length is None else max_length - 1

        # An empty string tokenizes to nothing, which would leave a marker-only document.
        encodings = self.tokenizer([passage or " " for passage in text], add_special_tokens=False)["input_ids"]
        sequences = [[marker_ids["document"]] + list(ids)[:content_limit] for ids in encodings]
        return self._pad_sequences(sequences, padding=padding, max_length=max_length, return_tensors=return_tensors)

    def process_images(self, images: ImageInput, **kwargs) -> BatchFeature:
        """Patchify images and build the document layout with two-axis position ids."""
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
        batch["pixel_values"] = image_inputs["pixel_values"]  # (num_patches, 3 * patch_size ** 2)
        batch["image_grid_hw"] = image_inputs["image_grid_hw"]  # (batch_size, 2)
        return batch

    def score_retrieval(
        self,
        query_embeddings: torch.Tensor | list[torch.Tensor],
        passage_embeddings: torch.Tensor | list[torch.Tensor],
        batch_size: int = 128,
        output_dtype: torch.dtype | None = None,
        output_device: str | torch.device = "cpu",
        *,
        normalize: bool = True,  # TODO: remove if we don't ship MeanMaxSim
    ) -> torch.Tensor:
        """Score query-passage pairs with MaxSim (multi-vector) or cosine similarity (dense).

        Representation is inferred from rank: 3-D / list-of-2-D grids use MaxSim; 2-D dense vectors
        use cosine. Both sides must use the same representation.

        Args:
            query_embeddings (`torch.Tensor` or `list[torch.Tensor]`):
                Multi-vector grids of shape `(num_queries, query_length, dim)` / list of
                `(query_length_i, dim)`, or dense vectors of shape `(num_queries, dim)`.
            passage_embeddings (`torch.Tensor` or `list[torch.Tensor]`):
                Same conventions as `query_embeddings`, for passages.
            batch_size (`int`, *optional*, defaults to 128):
                Chunk size over queries and passages when computing MaxSim (ignored for dense).
            output_dtype (`torch.dtype`, *optional*):
                Dtype of the returned score tensor. Defaults to the dtype of the computed scores.
            output_device (`str` or `torch.device`, *optional*, defaults to `"cpu"`):
                Device of the returned score tensor.
            normalize (`bool`, *optional*, defaults to `True`):
                MaxSim only. If `True`, divide each score by the number of non-padding query tokens
                (scores stay roughly in `[-1, 1]`). If `False`, return the raw ColBERT sum, which
                scales with query length — matching ColPali/ColQwen2 `score_retrieval`. Keyword-only
                for compatibility with those processors. Does not apply to dense cosine scoring, and
                is unrelated to L2-normalizing the embedding vectors (always done for MaxSim).

        Returns:
            `torch.Tensor` of shape `(num_queries, num_passages)`.
        """
        if len(query_embeddings) == 0 or len(passage_embeddings) == 0:
            raise ValueError("Both `query_embeddings` and `passage_embeddings` must be non-empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        if self._is_multi_vector(passage_embeddings):
            scores = self._maxsim_in_blocks(query_embeddings, passage_embeddings, batch_size, normalize)
        else:
            queries = self._as_dense(query_embeddings)  # (num_queries, dim)
            passages = self._as_dense(passage_embeddings)  # (num_passages, dim)
            scores = (
                torch.nn.functional.normalize(queries.float(), dim=-1)
                @ torch.nn.functional.normalize(passages.float(), dim=-1).t()
            )  # (num_queries, num_passages)

        return scores.to(output_dtype or scores.dtype).to(output_device)

    def _marker_ids(self) -> dict[str, int]:
        """Resolve marker token ids and validate that the tokenizer defines them."""
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
        grid = np.full(
            (grid_height, grid_width + 1), marker_ids["image"], dtype=np.int64
        )  # (grid_height, grid_width + 1)
        grid[:, grid_width] = marker_ids["row"]
        ids = [marker_ids["document"], marker_ids["image"], *grid.ravel().tolist()]

        positions = np.empty((len(ids), 2), dtype=np.int64)  # (sequence_length, 2)
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
            # Index 0 is the M-RoPE row axis, index 1 the column axis.
            grid = np.zeros((2, len(sequences), length), dtype=np.int64)  # (2, batch_size, sequence_length)
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
        """Filter text kwargs to the subset supported by this processor."""
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

    def _maxsim_in_blocks(
        self,
        query_embeddings: torch.Tensor | list[torch.Tensor],
        passage_embeddings: torch.Tensor | list[torch.Tensor],
        batch_size: int,
        normalize: bool,  # TODO: remove if we don't ship MeanMaxSim
    ) -> torch.Tensor:
        """Compute MaxSim scores in query-passage blocks."""
        query_grids, query_mask = _as_padded_grids(
            query_embeddings
        )  # (num_queries, query_length, dim), (num_queries, query_length)
        passage_grids, passage_mask = _as_padded_grids(
            passage_embeddings
        )  # (num_passages, passage_length, dim), (num_passages, passage_length)

        rows: list[torch.Tensor] = []
        for query_start in range(0, len(query_grids), batch_size):
            queries = slice(query_start, query_start + batch_size)
            columns = [
                maxsim_scores(
                    query_grids[queries],
                    passage_grids[passage_start : passage_start + batch_size],
                    query_mask[queries],
                    passage_mask[passage_start : passage_start + batch_size],
                    normalize=normalize,
                )
                for passage_start in range(0, len(passage_grids), batch_size)
            ]
            rows.append(torch.cat(columns, dim=1))
        return torch.cat(rows, dim=0)  # (num_queries, num_passages)

    def _is_multi_vector(self, embeddings: torch.Tensor | list[torch.Tensor]) -> bool:
        if isinstance(embeddings, torch.Tensor):
            return embeddings.dim() == 3
        return embeddings[0].dim() == 2

    def _as_dense(self, embeddings: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        return embeddings if isinstance(embeddings, torch.Tensor) else torch.stack(list(embeddings))  # (batch_size, dim)


__all__ = ["NeoMMEProcessor"]
