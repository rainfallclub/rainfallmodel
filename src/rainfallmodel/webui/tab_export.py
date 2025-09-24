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
import ast
import html


from ..common.packages import is_gradio_available
from ..common.resource import get_model_config
from ..common.resource import  get_tokenizer_config
from ..vocab.info import get_tokenizer
from .manager import Manager
from ..export.merge_lora import do_merge_lora
from ..common.resource import get_output_path

if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component



def create_format_tab(manager: "Manager") -> dict[str, "Component"]:
    input_elems = manager.get_base_elems()
    elem_dict = dict()
    # gr.Markdown("---")
    # gr.Markdown("#### 模型转换为Safetensors格式")
    # with gr.Row():
    #     model1_source_path = gr.Text(label="模型原始路径", value="",  interactive=True)
    # with gr.Row():
    #     model1_target_path = gr.Text(label="模型新路径", value="",  interactive=True, scale=8)
    #     model1_export_safetensors_btn = gr.Button(value="开始导出", variant="stop")
    gr.Markdown("---")
    gr.Markdown("#### 把LoRA合并到原始模型，生成新模型")

    with gr.Row():
        model_path_list = get_model_config()
        base_model2_source_path = gr.Dropdown(choices=model_path_list, label="模型路径或地址", value="rainfall_4m_base",  interactive=True, allow_custom_value=True)
    with gr.Row():
        lora_path = gr.Text(label="LoRA路径", value="",  interactive=True)
    with gr.Row():
        output_path_list = [get_output_path() + "_export"]
        model2_target_path = gr.Dropdown(choices=output_path_list, label="合并后的路径(可以选择，也可以自定义)",  interactive=True, allow_custom_value=True, scale=8)

        # model2_target_path = gr.Text(label="合并后的路径", value="",  interactive=True,  scale=8)
        export_model2_btn = gr.Button(value="开始合并", variant="stop")
    
    gr.Markdown("---")
    export_model2_btn.click(fn=do_merge_lora, inputs=[base_model2_source_path, lora_path, model2_target_path], outputs=[])

    return elem_dict


    