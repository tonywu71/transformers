<!--Copyright 2026 H Company and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-07-30.*

# NeoMME

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

NeoMME is a multilingual multimodal *encoder*: one bidirectional block stack reads text and document images
through the same layers. It is **vision-tower-free** — an image is turned into a row-major grid of flattened
`patch_size × patch_size` RGB patches that go straight into a small patch stem, so the number of image
tokens is decided by the resolution alone and no separate ViT is involved.

Architecturally it combines:

- factorized (ALBERT-style) token embeddings, with the masked-language-modeling decode tied through the same
  two matrices, so it costs no extra parameters;
- a sliding-window / global attention alternation where the **last layer is always global**, with two
  different sliding-window widths and per-layer-type RoPE (full rotary + short theta on sliding layers,
  partial rotary + long theta on global layers);
- a two-axis interleaved partial M-RoPE: image patches carry `(row, column)` positions, text carries the
  same position on both axes, and the dims past `partial_rotary_factor` stay position-blind;
- bidirectional grouped-query attention with parameter-free QK-norm, a sigmoid output gate and Exclusive
  Self-Attention, plus a squared-ReLU MLP and a learnable per-layer mix of the residual stream with the
  initial embedding stream.

Two checkpoints are released per size. The pretrained one is a masked-diffusion reading model
([`NeoMMEModel`] / [`NeoMMEForMaskedLM`]); the retrieval one adds two heads on top of a single backbone pass
([`NeoMMEForRetrieval`]): a ColBERT-style multi-vector grid scored with MaxSim, and a latent-attention
pooled dense vector scored with cosine similarity.

## Usage

Queries and documents are opposite retrieval sides, so they are encoded in separate calls: queries get a
`<query>` marker plus ten `<mask>` expansion tokens, documents get a `<doc>` marker.

```python
import torch
from PIL import Image

from transformers import NeoMMEForRetrieval, NeoMMEProcessor


processor = NeoMMEProcessor.from_pretrained("Hcompany/neomme-retrieval")
model = NeoMMEForRetrieval.from_pretrained("Hcompany/neomme-retrieval", device_map="auto")

queries = ["What was the revenue in 2024?", "Who signed the agreement?"]
pages = [Image.open("page_1.png"), Image.open("page_2.png")]

batch_queries = processor(text=queries, text_role="query").to(model.device)
batch_pages = processor(images=pages).to(model.device)

with torch.inference_mode():
    query_embeddings = model(**batch_queries).multivector_embeddings
    page_embeddings = model(**batch_pages).multivector_embeddings

scores = processor.score_retrieval(query_embeddings, page_embeddings)
print(scores)
```

`forward` returns both heads by default. Pass `output_dense=False` or `output_multivector=False` to compute
only one, and `dense_dim=...` to get a Matryoshka-truncated dense vector (truncated *before* it is
L2-normalized, so the prefix is itself a unit vector).
<!-- TODO: remove if we don't ship Matryoshka -->

## Notes

- `position_ids` has shape `(2, batch_size, sequence_length)` — index 0 is the M-RoPE row axis, index 1 the
  column axis. A plain `(batch_size, sequence_length)` tensor is accepted and expanded onto both axes, which
  is what text-only inputs want. Every image restarts its position grid at `(0, 0)`.
- `pixel_values` has shape `(num_patches, 3 * patch_size ** 2)`: flattened RGB patches for the whole batch,
  concatenated in batch order and row-major within an image, scaled to `pixel / 127.5 - 1`.
- `sliding_window_short` and `sliding_window_long` are **half-widths**: the attention band is
  `abs(i - j) <= window`.
- The image processor does not resize by default, so the patch grid tracks the image's native resolution.
  Pass `max_side`, `max_pixels` or `min_pixels` to bound the number of image tokens.
- Every norm in the backbone is parameter-free; only the patch stem's `LayerNorm` and the retrieval
  pooler's `RMSNorm` carry weights.

## NeoMMEConfig

[[autodoc]] NeoMMEConfig

## NeoMMEImageProcessor

[[autodoc]] NeoMMEImageProcessor
    - preprocess

## NeoMMEImageProcessorPil

[[autodoc]] NeoMMEImageProcessorPil
    - preprocess

## NeoMMEProcessor

[[autodoc]] NeoMMEProcessor
    - __call__
    - process_queries
    - process_text_documents
    - process_images
    - score_retrieval

## NeoMMEModel

[[autodoc]] NeoMMEModel
    - forward

## NeoMMEForMaskedLM

[[autodoc]] NeoMMEForMaskedLM
    - forward

## NeoMMEForRetrieval

[[autodoc]] NeoMMEForRetrieval
    - forward
