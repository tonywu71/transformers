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

<!-- TODO: write Overview -->

## Usage

<!-- TODO: write Usage -->

```python
# TODO: write usage example
```

## Notes

<!-- TODO: write Notes -->

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
