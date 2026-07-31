NeoMME is a vision-tower-free multilingual multimodal encoder: text tokens and image patches go through the same bidirectional trunk, so there is no separate ViT. Patches enter as flat pixel vectors through a small MLP stem, and positions come from a two-axis interleaved partial M-RoPE, so a patch grid keeps its 2-D structure while text stays a single axis. This port makes the released checkpoint loadable with `AutoModel` and ships the two retrieval heads it was trained with: a multi-vector late-interaction head (MaxSim) and a dense pooled head (cosine).

The research repo owns the checkpoint format, so the conversion script lives there (`hcompai/neomme`, `scripts/tooling/convert_neomme_weights_to_hf.py`) rather than under `src/transformers/models/neomme/` where other models keep theirs. That repo also holds the numerical parity gate: hcompai/neomme#107.

### Minimal example

```python
processor = AutoProcessor.from_pretrained(model_id)
model = NeoMMEForRetrieval.from_pretrained(model_id).eval()

queries = processor(text=["a chart of quarterly revenue"], text_role="query")
pages = processor(images=[page_image])

query_embeddings = model(**queries).multivector_embeddings
page_embeddings = model(**pages).multivector_embeddings
scores = processor.score_retrieval(query_embeddings, page_embeddings)
```

A single forward returns both heads, and `score_retrieval` picks MaxSim or cosine from the rank of what it is given.

### Parity

Checked against the research implementation on the 250M retrieval weights: multi-vector max abs diff 4.88e-07, dense 1.19e-07, and every converted tensor bit-identical.

The synthetic parity tests that run without Hub access used to prove less than they looked like. NeoMME starts every residual branch at exactly zero — `o_proj`, `down_proj`, the XSA `alpha`, the value-embedding table — mirroring the research init, which made a config-built model a bitwise identity on the embedding stream: scrambling `q_proj`, `kv_proj` and `up_proj` left `last_hidden_state` unchanged to the last bit. Both this suite and the parity gate now give those tensors standard fan-in init before comparing. Swapping K and V out of the fused `kv_proj` is caught by four of the five parity tests; before, it was caught by none.

> [!NOTE]
> The `@slow` integration tests currently point at a private staging repo and need Hub credentials; they get repointed at the public checkpoint before review. `check_config_docstrings` stays red for the same reason — `@auto_docstring(checkpoint=...)` needs an id that resolves publicly.

> [!NOTE]
> `sky_neomme_gpu_tests.yaml` and `.skyignore` are branch-local working files that run the GPU-only tests, and get dropped before this goes upstream. The running log of how the port deviates from its plan moved to the research repo, where the rest of the porting docs live: [`docs/transformers_integration/port_deviations.md`](https://github.com/hcompai/neomme/blob/transformers-implementation/docs/transformers_integration/port_deviations.md).

## Changes

<details>
<summary>Click here to expand</summary>

- **`NeoMMEConfig`** — `layer_types` keeps the standard two values while the two SWA half-widths (256/1024, alternating by sliding-layer parity) come from a derived `layer_window_sizes` property; `rope_parameters` is keyed by the layer types the model actually has, and a flat `rope_theta` reaches all of them
- **Only the released architecture is expressible** — `use_xsa`, `depth_scale` and `patch_stem` were research ablation switches whose research defaults no released NeoMME uses, so they are fixed here and refused by name rather than left as knobs that can only break the model; the config also carries just the three token ids the forward reads, and one encoding for the window widths. 31 fields down to 24
- **`NeoMMEModel` / `NeoMMEForRetrieval`** written as `modular_neomme.py` — factorized embeddings with a tied decode, GQA with parameter-free QK-norm before RoPE, XSA, squared-ReLU MLP, `x0` residual mix, and value embeddings on the first and last global layers
- **Mask construction** via `create_bidirectional_mask` with a per-width sliding overlay, one mask per distinct window rather than per layer type
- **`NeoMMEImageProcessor`** (PIL backend, no torchvision) emitting flat `pixel_values` plus `image_grid_hw`, and **`NeoMMEProcessor`** with the marker/query-expansion convention and `score_retrieval`
- **Tests** on `ModelTesterMixin`, `ImageProcessingTestMixin` and `ProcessorTesterMixin` (kwargs tests rewritten for one modality per call, the ColQwen2 pattern), plus integration tests against real weights
- **RoPE scaling takes effect instead of being ignored** — each layer type resolves its frequencies through its own `ROPE_INIT_FUNCTIONS` entry, so `rope_type="linear"` / `"dynamic"` / `"yarn"` are no longer accepted and dropped. The default path keeps the `theta ** -x` form the released checkpoints were built with, verified bit-identical; upstream's `1.0 / theta ** x` differs by one ULP, worth 6.1e-5 on cos/sin at the end of a 16384 context. A `partial_rotary_factor` that would not rotate a multiple of 4 dims is now refused rather than floored — at the tester's old `head_dim=8` that floor left global layers rotating **zero** dims, so global-layer RoPE was never exercised
- **Fixes from a cross-review** — resizing token embeddings now carries the second vocab-indexed table (the value embeddings) instead of raising on the first new id; a homogeneous `layer_types` is constructible; `get_dense_embeddings` takes the inputs first and validates `dense_dim`; a tokenizer whose marker tokens are absent is refused rather than silently encoding `<unk>`; the image path compiles with `fullgraph=True`; the processor honours `padding` / `return_tensors` instead of dropping them and raises on kwargs it does not implement; the `min_pixels` floor no longer discards `max_side` / `max_pixels`, matching hcompai/neomme#110 so the two resize paths stay identical

</details>
