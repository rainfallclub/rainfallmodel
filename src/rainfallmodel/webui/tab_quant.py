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
from ..common.resource import get_model_config
from ..common.resource import  get_tokenizer_config
from ..vocab.info import get_tokenizer
from .manager import Manager
from ..export.merge_lora import do_merge_lora
from ..common.resource import get_output_path
from ..quant.quant_bnb import do_quant_bnb_interface

if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component



def create_quant_tab(manager: "Manager") -> dict[str, "Component"]:
    input_elems = manager.get_base_elems()
    elem_dict = dict()

    gr.Markdown("---")
    with gr.Row():
        model_path_list = get_model_config()
        model_path = gr.Dropdown(choices=model_path_list, label="模型路径或地址(可以选择，也可以自定义)", value="rainfall_4m_base",  interactive=True, allow_custom_value=True)
   
    with gr.Row():
        output_path_list = [get_output_path() + "_quant"]
        output_dir = gr.Dropdown(choices=output_path_list, label="量化后的路径(可以选择，也可以自定义)",  interactive=True, allow_custom_value=True)
        quant_type_list = ['8bit', '4bit']
        quant_type = gr.Dropdown(choices=quant_type_list, label="量化类型", value='8bit',  interactive=True, allow_custom_value=True)
        # model2_target_path = gr.Text(label="合并后的路径", value="",  interactive=True,  scale=8)
    with gr.Row():
        export_model2_btn = gr.Button(value="开始量化", variant="stop")
    
    export_model2_btn.click(fn=do_quant_bnb_interface, inputs=[model_path, output_dir, quant_type], outputs=[])

    return elem_dict


    