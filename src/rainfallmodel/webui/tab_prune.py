# Copyright 2025 the LlamaFactory team. and the RainFallModel team.
#
# This code is inspired by the LlamaFactory.
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/webui/components/train.py
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

from typing import TYPE_CHECKING


from ..common.packages import is_gradio_available
from .manager import Manager
from ..common.resource import get_dataset_config, get_model_config, get_output_path

if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

def create_prune_tab(manager: "Manager") -> dict[str, "Component"]:
    """
    先做一个简易版的剪枝
    """
    input_elems = manager.get_base_elems()
    elem_dict = dict()

    gr.Markdown("---") 
    gr.HTML("<h3><center>模型剪枝这块内容涵盖太广，目前仅支持最简单的非结构化剪枝，后续再完善</center></h3>")
    gr.Markdown("---") 
    

    return elem_dict
