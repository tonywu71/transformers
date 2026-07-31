# NeoMME port — deviations from `docs/release/transformers_integration_plan.md`

Running log kept while implementing the `add-neomme` branch. Every entry is a place where the
implementation differs from the plan (or resolves one of its open questions), with the reason.

> This file lives on the `add-neomme` branch for convenience and is committed separately
> (`docs: …`) so it can be dropped with a single revert before the branch is published upstream.

## Resolved open questions (plan §11)

| Question | Resolution | Why |
|---|---|---|
| Public `position_ids` shape | **`(2, B, L)`** (axis-major: `[0]` = row, `[1]` = col). A plain `(B, L)` tensor is accepted and expanded to both axes (text-only case). | Matches the Qwen2-VL idiom of "leading dim = RoPE axis" instead of inventing a trailing-dim layout, and keeps the text path indistinguishable from any other encoder (`position_ids=(B, L)` just works). |
| Fused `kv_proj` kept vs split | **Kept fused.** | Default of `transformers_weight_mapping.md`; keeps the conversion a pure rename. The split transform stays documented if PR review forces it. |
| `MultiVectorMask` skiplist | Not yet reached (deliverable 2). | — |

## Deviations

### 1. `global_attn_every_n_layers` is 6, not 8

Plan §7.9 illustrates the layer pattern with "17 layers, N=8 → globals at {7, 15, 16}". The actual
release config (`configs/p1_pretraining_large256m.toml`) uses `global_attn_every_n_layers = 6`, so
the released 256M model has globals at **{5, 11, 16}** and 14 SWA layers. The *rule*
(`(i + 1) % N == 0 or i == last`) is what the port implements; only the plan's worked example was
stale. Config defaults in `configuration_neomme.py` follow the real release values.

### 2. `layer_types` carries only two values; window widths live in a separate list

`transformers` `layer_types` is a two-valued vocabulary (`full_attention` / `sliding_attention`) and
`rope_parameters` is keyed by exactly those values. NeoMME needs **two different SWA half-widths**
(short 256 / long 1024) alternating by cheap-layer parity, which does not fit `layer_types`.

Resolution: `layer_types` keeps the standard two values (so `rope_parameters` and every generic
utility keep working), and the per-layer band width is exposed as the derived
`NeoMMEConfig.layer_window_sizes` property (`None` on global layers). The model builds one mask per
*distinct* window instead of one per `layer_type`.

### 3. SWA masks are built with `create_bidirectional_mask(..., and_mask_function=...)`

`create_bidirectional_sliding_window_mask` reads a single `config.sliding_window`, which cannot
express two widths. The port instead composes the padding mask with
`sliding_window_bidirectional_overlay(w)` (`abs(q - kv) <= w`, i.e. exactly our half-width
convention) via `and_mask_function`. No new `config.sliding_window` field is introduced, so nothing
else in the library can silently build a wrong-width mask.

### 4. `unmask_empty_rows` keeps padding rows finite

Query rows at padding positions can end up fully masked once the SWA band is intersected with the
padding mask (a short row in a long batch), which is a NaN source in SDPA. `unmask_empty_rows` is a
post-hoc fixup on the built mask: it finds rows with no reachable key and opens them to *everything*,
rather than to themselves. A diagonal `or_mask_function` (`q_idx == kv_idx`) would be the tighter
construction, but it is not what the code does — the two differ in what a padding row attends to, and
only in what a padding row attends to, since a real query always reaches at least itself.

Either way the values are discarded: downstream, the multi-vector head *overwrites* padding rows with
`masked_fill` instead of multiplying by the mask, so a NaN could not leak even if it appeared
(`0 * NaN == NaN`).

### 5. No fast (torchvision) image processor

Plan pitfall §7.17 forbids torchvision. `NeoMMEImageProcessor` is therefore a `PilBackend`
processor and lives in `image_processing_pil_neomme.py` (the filename is what registers the `pil`
backend in `IMAGE_PROCESSOR_MAPPING_NAMES`; the class keeps the plain `NeoMMEImageProcessor` name
because it is the only backend NeoMME ships). PIL resizing is also what the research `patchify`
uses, so this is the only way to get bit-exact processor parity.

**Parity caveat:** research `patchify` calls `PIL.Image.draft()` for JPEG sources before resizing (a
libjpeg DCT-domain fast path that *changes pixel values*). An HF image processor receives an
already-decoded image and cannot do this. With the release default (no `max_side` / `max_pixels`,
native resolution ⇒ no resize at all) both paths are bit-identical; parity tests must therefore run
at native resolution, and any downscaling comparison is approximate.

### 6. Attention pooler keeps `nn.MultiheadAttention`

Per `transformers_weight_mapping.md` the default is to keep the module so the head conversion is a
pure rename (`attn_pooler → pooler`). `_init_weights` gains an explicit branch for it so the
"every parameter is initialized" test suite passes. If PR review rejects
`nn.MultiheadAttention`, the documented q/k/v split transform moves into the conversion script.

### 7. `NeoMMEForMaskedLM` exposes no output embeddings

The decode is fully tied through the factorized embedding (`h @ embedding_projection.weight @
word_embeddings.weight.T`), so there is no `lm_head` module. `get_output_embeddings()` returns
`None`, and no `_tied_weights_keys` entry exists.

### 8. The plan's `global_attn_every_n_layers` example is stale (see also #1)

The 256M release uses `global_attn_every_n_layers = 6`, so `NeoMMEConfig`'s default is 6 and the
released model has globals at {5, 11, 16}. The plan's "N=8 → {7, 15, 16}" was illustrative only.

### 9. `document_token_id` added to the config

`<img>` serves double duty in the layout `<doc> <img> <patch>… <row> …`: the occurrence right after
`<doc>` is a marker announcing an image document, every later one is a patch placeholder — and they
share one token id. The research repo sidesteps this with an explicit `patch_index` tensor, which the
public padded interface does not have. The model therefore needs to know the `<doc>` id to tell the
two apart, so `NeoMMEConfig` carries `document_token_id` (frozen id 5). The conversion script defaults
it to 5 (the research config does not store it) and validates it against the tokenizer when one is
supplied.

Going the other way, the plan's other four ids are **not** config fields. `pad`, `<doc>` and `<img>` are
the only ones the forward reads; `<query>`, `<row>`, `<mask>` and `<eos>` are laid out by
`NeoMMEProcessor`, which resolves them through the tokenizer, so keeping them here would duplicate
tokenizer state that nothing reads back.

### 10. One rotary module with per-layer-type buffers, not `rotary_emb` + `rotary_emb_local`

`NeoMMERotaryEmbedding` registers one `inv_freq` buffer per `layer_type` (the ModernBERT/Gemma3
idiom) instead of the research repo's two separate module instances. The buffers are non-persistent,
so no state-dict key is affected.

Each layer type's frequencies come from its own `rope_parameters["rope_type"]`, resolved through
`ROPE_INIT_FUNCTIONS` exactly as `Gemma4UnifiedTextRotaryEmbedding` does, so `rope_type="linear"`,
`"dynamic"` and `"yarn"` all work rather than being silently ignored. Two consequences:

- **`compute_default_rope_parameters` is written `theta ** -x`**, not the `1.0 / theta ** x` the scaled
  variants use. `ROPE_INIT_FUNCTIONS` has no `"default"` entry, so every model brings its own, and this
  is the form the research implementation
  ([`modeling_neomme.py`](https://github.com/hcompai/neomme/blob/main/neomme/model/modeling_neomme.py))
  built every released checkpoint's frequencies with. The two differ by one fp32 ULP; switching to
  upstream's form would move cos/sin by 6.1e-5 at the end of a 16384 context, 61% of the conversion
  parity gate's 1e-4 budget, for no behavioural gain.
- **`test_model_rope_scaling_frequencies` is skipped** for that ULP. Its per-layer-type
  `ntk_inv_freq <= original_inv_freq` check reads a layer type the test never forwards, so both sides
  are init-time values from the two different expressions. The three
  `test_model_rope_scaling_from_config` variants, which check that scaling changes the output, run.

### 10b. A `partial_rotary_factor` that does not rotate a multiple of 4 dims is rejected

The two M-RoPE axes consume frequencies in alternating pairs, so the rotating dims have to divide by 2
twice. `get_rotary_dim` used to floor to a multiple of 4, which built a working model whose spectrum was
narrower than its `config.json` advertised — at `head_dim=8` with the default factor of 0.25 it rotated
**zero** dims, which is what the tiny test config was doing before this check existed. `NeoMMEConfig`
now raises instead, and names the nearest usable factors.

### 11. Value embeddings are skipped on an `inputs_embeds`-only call

The value-embedding table is a per-token lookup, so it needs `input_ids`. Calling the model with
`inputs_embeds` alone therefore runs without the correction (the same limitation every VLM has for
its image scatter). Documented in the forward's docstring.

`test_inputs_embeds_matches_input_ids` is skipped for this reason. It passed until the `_init_weights`
patch below (#13) made the table non-zero, at which point the two paths visibly disagree — which is the
honest result, not a regression.

### 12. Grad-carrying layer arguments are positional

`hidden_states`, `initial_hidden_states` and `value_embeds` are passed to each encoder layer
positionally. Reentrant gradient checkpointing only re-attaches the autograd graph for positional
inputs; passing the shared value-embedding tensor as a keyword argument made
`test_training_gradient_checkpointing_use_reentrant_true` backward through the embedding node twice.

### 13. Test-suite accommodations

- `tests/test_modeling_common.py` gains a `num_hidden_layers <= 4` exception for the two NeoMME test
  classes: 4 is the minimum that exercises both sliding-window widths, both RoPE spectra and the
  always-global last layer (same precedent as `Gemma3nTextModelTest`).
- `NeoMMEForRetrieval` is tested in its own class with `is_training=False`, like the Col\* models: it
  returns embeddings, not a loss.
- The tester runs `head_dim=16`, not the more usual 8: with the default `partial_rotary_factor` of 0.25 a
  global layer needs 16 dims to rotate 4 (see #10b), which is also the smallest width that exercises both
  M-RoPE axes.
- `test_all_params_have_gradient = False`, because the common batch is text-only and the vision stem
  legitimately receives no gradient from it. `test_patch_stem_receives_gradients_from_images` covers
  the stem instead.
- `_init_weights` is patched for the duration of each test class so `o_proj`, `down_proj`, the XSA
  `alpha` and the value-embedding table are born non-zero. Every residual branch starts at exactly zero
  by design, which makes a config-built model a bitwise identity on the embedding stream: scrambling
  `q_proj`/`kv_proj`/`up_proj` left `last_hidden_state` unchanged to the last bit, so every inherited
  output comparison held regardless of what the trunk computed. The same patch is why the resize tests
  are now enabled and why #11 became visible.
- The processor test is on `ProcessorTesterMixin`, with the kwargs tests rewritten for one modality per
  call and `test_processor_with_multiple_inputs` skipped — the ColQwen2 pattern, which refuses both
  modalities in one call for the same reason. Its components are built locally instead of downloaded,
  because the marker tokens have to sit at frozen ids and the only checkpoint that has them is private.

### 14. `make fix-repo` reformats unrelated files

Running it applies newer ruff rules across the whole repository (67 files untouched by this work).
Those hunks were reverted; only the NeoMME files, the doc TOC entry, the auto-mapping entries and
the one test-common exception are committed.

### 15. The conversion script lives in the research repo, not here

Plan §6.4 and §9 put `convert_neomme_weights_to_hf.py` in `src/transformers/models/neomme/`, which is
where every other model keeps its conversion script. It was moved to the research repo at
`scripts/tooling/` instead, at the user's direction: that repo owns the checkpoint format the script
reads. Expect a reviewer to ask where it went, and say so in the PR description.

### 16. The marker convention ships as a chat template

Plan §6.3 mentions `chat_template.jinja` only for "plain-AutoModel multimodal usage". It turns out to
be load-bearing for two reasons, so the conversion script now writes one into every converted repo
(`scripts/tooling/neomme_chat_template.jinja` in the research repo):

- Without it the convention lives only in `NeoMMEProcessor`, so anyone loading the repo with a plain
  tokenizer gets unmarked, unexpanded queries and quietly worse recall.
- It is the hook sentence-transformers v6 uses. ST introspects the template for a `task` variable and
  routes `task="query"` / `task="document"` through it. Its own `query_expansion` config cannot express
  our convention — `length` there is a pad target ("pad to 32"), not "append ten" — and the branch's
  docstring directs chat-template backbones to the template instead.

Verified to render ids identical to `MARKERS.query_seq` / `doc_seq`, and to agree with
`NeoMMEProcessor` on the same inputs.

### 17. The first real checkpoint broke four assumptions

Converting `Hcompany/neomme-retrieval` exposed gaps no synthetic fixture could, because the fixture was
built from the same `NeoMMEConfig` the script was written against:

- **Config schema drift.** The release export writes `late_dim`, `heads`, `with_*_head` and `ocr_*`,
  and no `embedding_dim`. The script's blocklist crashed on the unknown fields; it is now an allowlist
  over `NeoMMEConfig`'s own annotations, with a `late_dim -> embedding_dim` alias and dropped keys logged.
- **`cheap_mixer` is gone from newer exports** — while a stale `gdn_kind: "kda"` is still written. The
  GDN refusal now reads the weight keys, which cannot go stale.
- **`pad_token_id: None`.** It is filled from the tokenizer, matching `NeoMMEConfig.apply_token_ids`;
  only a genuine disagreement raises. Same for the other two ids the forward reads.
- **Saved dtype.** Instantiating the port materializes fp32, so the first conversion wrote 1069 MB from
  a 535 MB bf16 source. The saved dtype now follows the source, with a `--dtype` override.

Parity on the real weights: multi-vector 4.88e-07, dense 1.19e-07, all 151 tensors bit-identical.

### 18. The retrieval helpers take explicit arguments

`get_multivector_embeddings` / `get_dense_embeddings` originally took `**kwargs` only. A
sentence-transformers v6 `MultiVectorEncoder` dispatches to a model method by introspection, filtering
its feature dict against `inspect.signature(method).parameters` — which for a `**kwargs`-only signature
is `{"kwargs"}`, so every input including `input_ids` was dropped. The five real inputs are now named.

### 19. Integration tests point at a private staging repo

`Hcompany/neomme-250M-retrieval-dev-transformers-v0.3` is private, so the `@slow` integration tests
need Hub credentials. Repoint them at the public checkpoint at release. Their expected score matrices
were measured against the research implementation on the same weights.

The masked-LM test loads that same retrieval repo rather than the pretrained one: `NeoMMEForMaskedLM`
adds no parameters, so it takes the backbone through `base_model_prefix`. It pins the decode's
arithmetic, not reading quality, and should move to the pretrained checkpoint once that is converted.

### 20. GPU-only tests run on an ephemeral cluster

`test_flash_attn_2_inference_equivalence` and friends are already collected through `ModelTesterMixin`
but skip without a GPU. `sky_neomme_gpu_tests.yaml` provisions an H100 with 60-minute autodown to run
them; no new test code was needed.

That run does **not** by itself verify plan pitfall §7.21 (FA's inclusive `window_size` band agreeing
with our half-width convention, on both widths). The common test builds the model with
`model_class(config)`, and until the `_init_weights` patch in `test_modeling_neomme.py` landed that made
the whole trunk an identity, so eager and FA agreed trivially; the test also expects a
`config.sliding_window` to shrink, which NeoMME deliberately does not have. With live residual branches
the comparison is real, but the window widths themselves are pinned by
`test_every_layer_attends_bidirectionally_within_its_window` on CPU and by the numerical parity gate in
`hcompai/neomme`, not by the H100 run.
