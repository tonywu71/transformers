#           🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#               This file was automatically generated from <path_to_modular_file.py>.
#         Do NOT edit this file manually as any edits will be overwritten by the generation of
#         the file from the modular. If any change should be done, please apply the change to the
#                           modular_xxx.py file directly. One of our CI enforces this
#           🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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


from ..paligemma import (
    PaliGemmaConfig,
)


class ColPaliConfig(PaliGemmaConfig):
    r"""
    This is the configuration class to store the configuration of a [`ColPaliForRetrieval`]. It is used to instantiate an
    ColPaliForRetrieval according to the specified arguments, defining the model architecture.

    The ColPali config is stricly equivalent to the PaliGemma config, but with a different model type.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "colpali"
        self.is_composition = False