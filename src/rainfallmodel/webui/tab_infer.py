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


if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component


def create_infer_tab(manager: "Manager") -> dict[str, "Component"]:
    input_elems = manager.get_base_elems()
    elem_dict = dict()
    with gr.Row():
        model_path = gr.Textbox(label="模型路径", value="", interactive=True)
        dtype_list = ["auto", "bfloat16", "float16", "float32"]
        dtype = gr.Dropdown(choices=dtype_list, label="dtype类型", value="auto", interactive=True, visible=False)

    with gr.Row():
        load_model_btn = gr.Button(value="加载模型", variant="primary")
        unload_model_btn = gr.Button(value="卸载模型")

    info_box = gr.Textbox(value="模型状态: 模型未加载", interactive=False)

    input_elems.update({model_path, dtype})
    elem_dict.update(
        dict(
            model_path=model_path,
            dtype=dtype,
        )
    )

    load_model_btn.click(manager.infer_runner.load_model, inputs=input_elems, outputs=[info_box])
    unload_model_btn.click(manager.infer_runner.unload_model, inputs=[], outputs=[info_box])

    with gr.Column() as chat_box:
        chatbot = gr.Chatbot(type="messages", show_copy_button=True)
        messages = gr.State(value=[])
        with gr.Row():
            with gr.Column(scale=4):
                query = gr.Textbox(label="用户输入", lines=3)
                submit_btn = gr.Button(value="提交", variant="primary")

            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(label="最大token数", minimum=8, maximum=8192, value=32, step=1)
                top_p = gr.Slider(label="top-p值", minimum=0.01, maximum=1.0, value=0.7, step=0.01)
                temperature = gr.Slider(label="温度系数", minimum=0.01, maximum=1.5, value=0.1, step=0.01)
                clear_btn = gr.Button(value="清空")
    
    generate_inputs = [chatbot, messages, query, max_new_tokens, top_p, temperature]
    submit_btn.click(manager.infer_runner.interface_generate, generate_inputs, [chatbot, messages, query])
    clear_btn.click(lambda: ([], []), outputs=[chatbot, messages])

    return elem_dict
























