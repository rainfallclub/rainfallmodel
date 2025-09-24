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
from .tab_todo import create_todo_tab
from .tab_prune_unstructure import create_global_unstructure_prune_tab, create_local_unstructure_prune_tab

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

    with gr.Blocks():
        # pass
        with gr.Tab("非结构化全局剪枝"):
            base_dict = create_global_unstructure_prune_tab(manager)
            elem_dict.update(base_dict)
        with gr.Tab("非结构化局部剪枝"):
            base_dict = create_local_unstructure_prune_tab(manager)
            elem_dict.update(base_dict)
        # with gr.Tab("结构化剪枝"):
        #     compare_dict = create_todo_tab(manager)
        #     elem_dict.update(compare_dict)
    
    

    return elem_dict
