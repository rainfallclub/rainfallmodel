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
from ..common.resource import get_model_config


if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component


def create_infer_base_tab(manager: "Manager") -> dict[str, "Component"]:
    input_elems = manager.get_base_elems()
    elem_dict = dict()
    with gr.Row():
        model_list = get_model_config()
        base_model_path = gr.Dropdown(choices=model_list, label="模型路径(可以选择已有模型，也可以输入本地路径)", value="rainfall_4m_base", interactive=True, allow_custom_value=True)

        # base_model_path = gr.Textbox(label="模型路径", value="", interactive=True)
        base_dtype_list = ["auto", "bfloat16", "float16", "float32"]
        base_dtype = gr.Dropdown(choices=base_dtype_list, label="dtype类型", value="auto", interactive=True, visible=False)

    with gr.Row():
        load_model_btn = gr.Button(value="加载模型", variant="primary")
        unload_model_btn = gr.Button(value="卸载模型")

    info_box = gr.Textbox(value="模型状态: 模型未加载", interactive=False)

    input_elems.update({base_model_path, base_dtype})
    elem_dict.update(
        dict(
            base_model_path=base_model_path,
            base_dtype=base_dtype,
        )
    )

    load_model_btn.click(manager.infer_base_runner.load_model, inputs=input_elems, outputs=[info_box])
    unload_model_btn.click(manager.infer_base_runner.unload_model, inputs=[], outputs=[info_box])

    with gr.Column() as chat_box:
        base_chatbot = gr.Chatbot(type="messages", show_copy_button=True)
        base_messages = gr.State(value=[])
        with gr.Row():
            with gr.Column(scale=4):
                base_query = gr.Textbox(label="用户输入", lines=3)
                base_submit_btn = gr.Button(value="提交", variant="primary")

            with gr.Column(scale=1):
                base_max_new_tokens = gr.Slider(label="最大token数", minimum=8, maximum=8192, value=64, step=1)
                base_top_p = gr.Slider(label="top-p值", minimum=0.01, maximum=1.0, value=0.7, step=0.01)
                base_temperature = gr.Slider(label="温度系数", minimum=0.01, maximum=1.5, value=0.1, step=0.01)
                base_clear_btn = gr.Button(value="清空")
    
    generate_inputs = [base_chatbot, base_messages, base_query, base_max_new_tokens, base_top_p, base_temperature]
    base_submit_btn.click(manager.infer_base_runner.interface_generate, generate_inputs, [base_chatbot, base_messages, base_query])
    base_clear_btn.click(lambda: ([], []), outputs=[base_chatbot, base_messages])

    return elem_dict
























