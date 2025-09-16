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
from .tab_infer_base import create_infer_base_tab
from .tab_infer_chat import create_infer_chat_tab

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component


def create_infer_tab(manager: "Manager") -> dict[str, "Component"]:
    elem_dict = dict()
    with gr.Blocks():
        with gr.Tab("续写模式[刚完成预训练的模型]"):
            base_dict = create_infer_base_tab(manager)
            elem_dict.update(base_dict)
        with gr.Tab("聊天模式[以完成指令微调后的模型]"):
            chat_dict = create_infer_chat_tab(manager)
            elem_dict.update(chat_dict)
    return elem_dict  
    
























