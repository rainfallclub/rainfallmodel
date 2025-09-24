# Copyright 2025 the LlamaFactory team. and the RainFallModel team.
#
# This code is inspired by the LlamaFactory.
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/webui/runner.py
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



from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Optional
from ..common.packages import is_gradio_available
from .locales import ALERTS
from ..inference.infer_interface import InferInterface
from ..common.misc import torch_gc
from ..prune.unstructured_prune import do_unstructure_global_prune, do_unstructure_local_prune

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component
    from .manager import Manager

class PruneRunner:
    r"""A class to manage the running status."""

    def __init__(self, manager: "Manager") -> None:
        self.manager = manager   

    def comma_string_to_list(self, input_string:str) -> list:
        """
        将逗号分隔的字符串转换为列表
        """
        items = input_string.split(",")
        items = [item.strip() for item in items]
        items = [item for item in items if item]
        return items

    def do_unstructure_global_prune(self, data):
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        prune_conf = dict(
            unstructure_global_model_path = get("prune.unstructure_global_model_path"),
            unstructure_global_output_dir = get("prune.unstructure_global_output_dir"),
            unstructure_global_prune_rate = get("prune.unstructure_global_prune_rate"),
            unstructure_global_prune_method = get("prune.unstructure_global_prune_method"),
        )
        do_unstructure_global_prune(prune_conf)
        lang = self.manager.get_lang()
        gr.Info(ALERTS["prune_done"][lang])
    
    
    def do_unstructure_local_prune(self, data):
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        prune_conf = dict(
            unstructure_local_model_path = get("prune.unstructure_local_model_path"),
            unstructure_local_output_dir = get("prune.unstructure_local_output_dir"),
            unstructure_local_prune_method = get("prune.unstructure_local_prune_method"),
            unstructure_local_prune_layers = get("prune.unstructure_local_prune_layers"),

            unstructure_local_prune_self_attn_flag = get("prune.unstructure_local_prune_self_attn_flag"),
            unstructure_local_prune_self_attn_rate = get("prune.unstructure_local_prune_self_attn_rate"),
            unstructure_local_prune_self_attn_modules = get("prune.unstructure_local_prune_self_attn_modules"),

            unstructure_local_prune_mlp_flag = get("prune.unstructure_local_prune_mlp_flag"),
            unstructure_local_prune_mlp_rate = get("prune.unstructure_local_prune_mlp_rate"),
            unstructure_local_prune_mlp_modules = get("prune.unstructure_local_prune_mlp_modules"),
            
        )
        do_unstructure_local_prune(prune_conf)
        lang = self.manager.get_lang()
        gr.Info(ALERTS["prune_done"][lang])
    
        
        


    



        
        

    














