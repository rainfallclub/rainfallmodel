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
from ..common.resource import get_dataset_config, get_output_path


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


def create_vocab_train_tab(manager: "Manager") -> dict[str, "Component"]:
    input_elems = manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        dataset_path_list = get_dataset_config()
        dataset_path = gr.Dropdown(choices=dataset_path_list, label="数据集文件(支持本地文件，也可以下拉框中选择在线数据集)", allow_custom_value=True)

    with gr.Row():
        output_path_list = [get_output_path() + "_tokenizer"]
        output_dir = gr.Dropdown(choices=output_path_list, label="输出路径", allow_custom_value=True)

    with gr.Row():
        vocab_size_values = ["3200", "6400", "50257", "152064"]
        vocab_size = gr.Dropdown(choices=vocab_size_values, label="词表大小", value="6400", interactive=True, allow_custom_value=True)
        algorithms = ["bpe"] # 目前仅支持这一个
        algorithm_name = gr.Dropdown(choices=algorithms, label="分词算法", value="bpe", interactive=True, allow_custom_value=True)
        min_frequency = gr.Textbox(label="最低词频", value="2", interactive=True)
       
    input_elems.update({dataset_path, output_dir, vocab_size, algorithm_name, min_frequency})
    elem_dict.update(dict(dataset_path=dataset_path, output_dir=output_dir, vocab_size=vocab_size, algorithm_name=algorithm_name, min_frequency=min_frequency))



    with gr.Row():
        padding_token_values = ["<pad>", "<padding>"]
        padding_token = gr.Dropdown(choices=padding_token_values, label="填充token", value="<pad>", interactive=True, allow_custom_value=True)
        unknown_token_values = ["<unk>", "<unknown>"]
        unknown_token = gr.Dropdown(choices=unknown_token_values, label="未知token", value="<unk>", interactive=True, allow_custom_value=True)
        bos_token_values = ["<s>", "<bos>", "<|beginoftext|>", "<|im_start|>"]
        bos_token =  gr.Dropdown(choices=bos_token_values, label="开始token", value="<bos>", interactive=True, allow_custom_value=True)
        eos_token_values = ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"]
        eos_token =  gr.Dropdown(choices=eos_token_values, label="结束token", value="<eos>", interactive=True, allow_custom_value=True)

    input_elems.update({padding_token, unknown_token, eos_token, bos_token})
    elem_dict.update(
        dict(
            padding_token=padding_token,
            unknown_token=unknown_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )
    )

    with gr.Row():
        special_tokens = gr.Textbox(label="其他特殊token，用空格分开即可", lines=3)
    input_elems.update({special_tokens})
    elem_dict.update(
        dict(
            special_tokens=special_tokens
        )
    )



    with gr.Row():
        start_btn = gr.Button(value="开始训练", variant="primary")
       

    # todo，目前先搁置，等有时间了再实现
    # with gr.Column(scale=1):
    #     loss_viewer = gr.Plot()
    # with gr.Row():
    #     use_swanlab = gr.Checkbox()
    #     swanlab_project = gr.Textbox(value="rainfallmodel")
    #     swanlab_run_name = gr.Textbox()
    #     swanlab_workspace = gr.Textbox()
    #     swanlab_api_key = gr.Textbox()
    #     swanlab_mode = gr.Dropdown(choices=["cloud", "local"], value="cloud")
    #     swanlab_link = gr.Markdown(visible=False)

    with gr.Row():
        progress_bar = gr.Slider(visible=False, interactive=False)
    with gr.Row():
        output_box = gr.Markdown()

    elem_dict.update(
        dict(
            output_box=output_box,
            progress_bar=progress_bar,
        )
    )
    # output_elems = [output_box, progress_bar, loss_viewer, swanlab_link]
    output_elems = [output_box, progress_bar]
    start_btn.click(manager.vocab_runner.run_vocab_train, input_elems, output_elems)



    return elem_dict


    