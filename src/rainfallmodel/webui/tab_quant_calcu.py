



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
from ..quant.quant_calcu import do_quant_calcu

if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component



def create_quant_calcu_tab(manager: "Manager") -> dict[str, "Component"]:
    input_elems = manager.get_base_elems()
    elem_dict = dict()
    gr.Markdown("---")
    gr.Markdown("#### 量化和反量化演示")
    with gr.Row():
        matrix_row_value_list = [2, 3, 4]
        matrix_row_value = gr.Dropdown(choices=matrix_row_value_list, label="矩阵的行", value="2",  interactive=True, allow_custom_value=True)
        matrix_col_value_list = [2, 3, 4]
        matrix_col_value = gr.Dropdown(choices=matrix_col_value_list, label="矩阵的列", value="3",  interactive=True, allow_custom_value=True)
        matrix_value_scale_list = [1, 10, 100]
        matrix_value_scale = gr.Dropdown(choices=matrix_value_scale_list, label="数据缩放系数", value="1",  interactive=True, allow_custom_value=True)
    with gr.Row():
        export_model2_btn = gr.Button(value="数据演示", variant="stop")
    with gr.Row():
        calculate_result_content = gr.Markdown()
    
    export_model2_btn.click(fn=do_quant_calcu, inputs=[matrix_row_value, matrix_col_value, matrix_value_scale], outputs=[calculate_result_content])

    return elem_dict






















