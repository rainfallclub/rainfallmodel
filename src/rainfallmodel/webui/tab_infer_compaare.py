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
from .locales import ALERTS
from ..common.resource import get_model_config

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component


def create_infer_compare_tab(manager: "Manager") -> dict[str, "Component"]:
    input_elems = manager.get_base_elems()
    elem_dict = dict()

    model_list = get_model_config()
    with gr.Row():
        compare_teacher_model_path = gr.Dropdown(choices=model_list, label="教师模型路径(可以选择已有模型，也可以输入本地路径)", value="minimind_v2", interactive=True, allow_custom_value=True)
    with gr.Row():
        compare_student_model_path = gr.Dropdown(choices=model_list, label="学生模型路径(可以选择已有模型，也可以输入本地路径)", value="rainfall_4m_base", interactive=True, allow_custom_value=True)


    with gr.Row():
        load_model_btn = gr.Button(value="加载模型", variant="primary")
        unload_model_btn = gr.Button(value="卸载模型")

    
    lang = manager.get_lang()
    model_status = ALERTS["infer_backend_unload"][lang]  # 读取模型状态失败
    # 暂时不处理状态
    # if manager.infer_chat_runner.loaded:
    #     model_status = ALERTS["infer_backend_loaded"][lang]
    # else:
    #     model_status = ALERTS["infer_backend_unload"][lang]

    info_box = gr.Textbox(model_status, interactive=False)

    input_elems.update({compare_teacher_model_path, compare_student_model_path})
    elem_dict.update(
        dict(
            compare_teacher_model_path=compare_teacher_model_path,
            compare_student_model_path=compare_student_model_path,
        )
    )

    load_model_btn.click(manager.infer_compare_runner.load_model, inputs=input_elems, outputs=[info_box])
    unload_model_btn.click(manager.infer_compare_runner.unload_model, inputs=[], outputs=[info_box])

    with gr.Column() as chat_box:
        chatbot = gr.Chatbot(type="messages", show_copy_button=True)
        messages = gr.State(value=[])
        with gr.Row():
            with gr.Column(scale=4):
                query = gr.Textbox(label="用户输入", lines=3)
                submit_btn = gr.Button(value="提交", variant="primary")

            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(label="最大token数", minimum=8, maximum=8192, value=64, step=1)
                top_p = gr.Slider(label="top-p值", minimum=0.01, maximum=1.0, value=0.7, step=0.01)
                temperature = gr.Slider(label="温度系数", minimum=0.01, maximum=1.5, value=0.1, step=0.01)
                clear_btn = gr.Button(value="清空")
    
    generate_inputs = [chatbot, messages, query, max_new_tokens, top_p, temperature]
    submit_btn.click(manager.infer_compare_runner.interface_chat, generate_inputs, [chatbot, messages, query])
    clear_btn.click(lambda: ([], []), outputs=[chatbot, messages])

    return elem_dict
























