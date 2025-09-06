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


from ..common.packages import is_gradio_available
from ..common.resource import  get_tokenizer_config
from ..vocab.info import get_tokenizer
from .manager import Manager


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

tokenizer = None

def show_tokenizer_info_click(tokenizer_path):
    tokenizer = get_tokenizer(tokenizer_path)
    vocab_info = str(tokenizer)
    return "词表详情:" + vocab_info

def clear_tokenizer_info_click():
    return ""

def do_tokenizer_encode_click(input_str, tokenizer_path):
    tokenizer = get_tokenizer(tokenizer_path)
    return "编码后的内容: " + str(tokenizer.encode(input_str))

def do_tokenizer_decode_click(input_ids, tokenizer_path):
    tokenizer = get_tokenizer(tokenizer_path)
    input_ids_list = ast.literal_eval(input_ids)
    decoded_str = tokenizer.decode(input_ids_list)
    return "解码后的内容: " + decoded_str

def create_vocab_info_tab(manager: "Manager") -> dict[str, "Component"]:
    input_elems = manager.get_base_elems()
    elem_dict = dict()
    with gr.Row():
        tokenizer_path_list = get_tokenizer_config()
        tokenizer_path = gr.Dropdown(choices=tokenizer_path_list, label="词表文件",  interactive=True, allow_custom_value=True)
    
    with gr.Row():
        to_encoded_str = gr.Text(label="待编码内容", value="雨落大模型",  interactive=True)

    with gr.Row():
        to_decoded_str = gr.Text(label="待解码内容", value="[4224, 3052, 611, 2933]",  interactive=True)
    
    # tokenizer = None
    with gr.Row():
        vocab_info_btn = gr.Button(value="查看词表详情", variant="stop")
        vocab_clean_btn = gr.Button(value="清空词表详情", variant="primary")
        vocab_encode_btn = gr.Button(value="对内容进行编码", variant="secondary")
        vocab_decode_btn = gr.Button(value="对内容进行解码", variant="huggingface")

    
    with gr.Row():
        encoded_content = gr.Markdown()
    with gr.Row():
        decoded_content = gr.Markdown()
    with gr.Row():
        vocab_info_json = gr.Markdown()

    vocab_info_btn.click(fn=show_tokenizer_info_click, inputs=[tokenizer_path], outputs=[vocab_info_json])
    vocab_clean_btn.click(fn=clear_tokenizer_info_click, inputs=[], outputs=[vocab_info_json])
    vocab_encode_btn.click(fn=do_tokenizer_encode_click, inputs=[to_encoded_str, tokenizer_path], outputs=[encoded_content])
    vocab_decode_btn.click(fn=do_tokenizer_decode_click, inputs=[to_decoded_str, tokenizer_path], outputs=[decoded_content])

    return elem_dict


    