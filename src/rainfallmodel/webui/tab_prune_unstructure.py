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
from ..common.resource import get_dataset_config, get_model_config, get_output_path
from .tab_todo import create_todo_tab
from ..prune.prune_calcu import do_prune_calcu

if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

def create_global_unstructure_prune_tab(manager: "Manager") -> dict[str, "Component"]:
    """
    非结构化剪枝
    """
    input_elems = manager.get_base_elems()
    elem_dict = dict()
    gr.Markdown("---") 
    gr.Markdown("#### 资源配置")
    with gr.Row():
        model_path_list = get_model_config()
        unstructure_global_model_path = gr.Dropdown(choices=model_path_list, label="模型地址", value="rainfall_4m_base",  interactive=True, allow_custom_value=True)
    with gr.Row():
        output_path_list = [get_output_path() + "_uns_global_prune"]
        unstructure_global_output_dir = gr.Dropdown(choices=output_path_list, label="输出路径",  interactive=True, allow_custom_value=True)

    input_elems.update({unstructure_global_model_path, unstructure_global_output_dir})
    elem_dict.update(
        dict(
            unstructure_global_model_path=unstructure_global_model_path, 
            unstructure_global_output_dir=unstructure_global_output_dir,
        )
    )
    gr.Markdown("#### 剪枝配置")
    with gr.Row():     
        unstructure_global_prune_rate_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        unstructure_global_prune_rate = gr.Dropdown(choices=unstructure_global_prune_rate_list, label="剪枝比例",  value=0.1, interactive=True, allow_custom_value=True)
        prune_method_list = ["L1Unstructured", "RandomStructured"]
        unstructure_global_prune_method = gr.Dropdown(choices=prune_method_list, label="剪枝方法",  value="L1Unstructured", interactive=True, allow_custom_value=True)
        # unstructure_global_prune_modules = gr.Text(label="剪枝模块(用半角逗号分开)", value="self_attn,mlp",  interactive=True)
        # start_btn.click()
    
    
    input_elems.update({unstructure_global_prune_rate, unstructure_global_prune_method})
    elem_dict.update(
        dict(
            unstructure_global_prune_rate=unstructure_global_prune_rate, 
            unstructure_global_prune_method=unstructure_global_prune_method,
            # unstructure_global_prune_modules=unstructure_global_prune_modules
        )
    )
    
    with gr.Row():
        start_btn = gr.Button(value="开始剪枝", variant="primary")

    start_btn.click(manager.prune_runner.do_unstructure_global_prune, inputs=input_elems, outputs=[])
    return elem_dict

def create_local_unstructure_prune_tab(manager: "Manager") -> dict[str, "Component"]:
    """
    非结构化剪枝
    """
    input_elems = manager.get_base_elems()
    elem_dict = dict()
    gr.Markdown("---") 
    gr.Markdown("#### 资源配置")
    with gr.Row():
        model_path_list = get_model_config()
        unstructure_local_model_path = gr.Dropdown(choices=model_path_list, label="模型地址", value="rainfall_4m_base",  interactive=True, allow_custom_value=True)
    with gr.Row():
        output_path_list = [get_output_path() + "_uns_local_prune"]
        unstructure_local_output_dir = gr.Dropdown(choices=output_path_list, label="输出路径",  interactive=True, allow_custom_value=True)

    input_elems.update({unstructure_local_model_path, unstructure_local_output_dir})
    elem_dict.update(
        dict(
            unstructure_local_model_path=unstructure_local_model_path, 
            unstructure_local_output_dir=unstructure_local_output_dir,
        )
    )

    gr.Markdown("---")
    gr.Markdown("#### 剪枝整体配置")
    with gr.Row():
        prune_method_list = ["L1Unstructured", "RandomStructured"]
        unstructure_local_prune_method = gr.Dropdown(choices=prune_method_list, label="剪枝方法",  value="L1Unstructured", interactive=True, allow_custom_value=True)
        unstructure_local_prune_layers_list = [1,2,4,8]
        unstructure_local_prune_layers = gr.Dropdown(choices=unstructure_local_prune_layers_list, label="剪枝层数",  value=4, interactive=True, allow_custom_value=True)
    
    input_elems.update({unstructure_local_prune_method, unstructure_local_prune_layers})
    elem_dict.update(
        dict(
            unstructure_local_prune_method=unstructure_local_prune_method, 
            unstructure_local_prune_layers=unstructure_local_prune_layers,
        )
    )    
    


    gr.Markdown("#### 注意力部分剪枝配置")
    with gr.Row():     
        unstructure_local_prune_self_attn_flag_list = [True, False]
        unstructure_local_prune_self_attn_flag = gr.Dropdown(choices=unstructure_local_prune_self_attn_flag_list, label="是否对注意力层剪枝",  value=True, interactive=True, allow_custom_value=True)
        unstructure_local_prune_rate_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        unstructure_local_prune_self_attn_rate = gr.Dropdown(choices=unstructure_local_prune_rate_list, label="剪枝比例",  value=0.1, interactive=True, allow_custom_value=True)
        unstructure_local_prune_self_attn_modules = gr.Text(label="剪枝模块(用半角逗号分开)", value="q_proj,k_proj,v_proj,o_proj",  interactive=True)
        # start_btn.click()
    
    
    input_elems.update({unstructure_local_prune_self_attn_flag, unstructure_local_prune_self_attn_rate, unstructure_local_prune_self_attn_modules})
    elem_dict.update(
        dict(
            unstructure_local_prune_self_attn_flag=unstructure_local_prune_self_attn_flag, 
            unstructure_local_prune_self_attn_rate=unstructure_local_prune_self_attn_rate,
            unstructure_local_prune_self_attn_modules=unstructure_local_prune_self_attn_modules
        )
    )

    gr.Markdown("#### 前馈神经网络部分剪枝配置")
    with gr.Row():     
        unstructure_local_prune_mlp_flag_list = [True, False]
        unstructure_local_prune_mlp_flag = gr.Dropdown(choices=unstructure_local_prune_mlp_flag_list, label="是否对前馈神经网络剪枝",  value=True, interactive=True, allow_custom_value=True)
        unstructure_local_prune_rate_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        unstructure_local_prune_mlp_rate = gr.Dropdown(choices=unstructure_local_prune_rate_list, label="剪枝比例",  value=0.1, interactive=True, allow_custom_value=True)
        unstructure_local_prune_mlp_modules = gr.Text(label="剪枝模块(用半角逗号分开)", value="gate_proj,up_proj,down_proj",  interactive=True)
        # start_btn.click()
    
    
    input_elems.update({unstructure_local_prune_mlp_flag, unstructure_local_prune_mlp_rate, unstructure_local_prune_mlp_modules})
    elem_dict.update(
        dict(
            unstructure_local_prune_mlp_flag=unstructure_local_prune_mlp_flag, 
            unstructure_local_prune_mlp_rate=unstructure_local_prune_mlp_rate,
            unstructure_local_prune_mlp_modules=unstructure_local_prune_mlp_modules
        )
    )
    
    with gr.Row():
        start_btn = gr.Button(value="开始剪枝", variant="primary")

    start_btn.click(manager.prune_runner.do_unstructure_local_prune, inputs=input_elems, outputs=[])
    return elem_dict



def create_prune_calcu_tab(manager: "Manager") -> dict[str, "Component"]:
    input_elems = manager.get_base_elems()
    elem_dict = dict()
    gr.Markdown("---")
    gr.Markdown("#### 剪枝简易演示")
    with gr.Row():
        matrix_row_value_list = [2, 3, 4]
        matrix_row_value = gr.Dropdown(choices=matrix_row_value_list, label="矩阵的行", value="2",  interactive=True, allow_custom_value=True)
        matrix_col_value_list = [2, 3, 4]
        matrix_col_value = gr.Dropdown(choices=matrix_col_value_list, label="矩阵的列", value="5",  interactive=True, allow_custom_value=True)
        matrix_value_scale_list = [1, 10, 100]
        matrix_value_scale = gr.Dropdown(choices=matrix_value_scale_list, label="数据缩放系数", value="1",  interactive=True, allow_custom_value=True)
        prune_method_list = ["L1Unstructured", "RandomStructured"]
        unstructure_prune_calcu_method = gr.Dropdown(choices=prune_method_list, label="剪枝方法",  value="L1Unstructured", interactive=True, allow_custom_value=True)
        unstructure_prune_amount_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        unstructure_prune_calcu_amount = gr.Dropdown(choices=unstructure_prune_amount_list, label="剪枝比例",  value=0.1, interactive=True, allow_custom_value=True)
        
    
    with gr.Row():
        export_model2_btn = gr.Button(value="数据演示", variant="stop")
    with gr.Row():
        calculate_result_content = gr.Markdown()
    
    export_model2_btn.click(fn=do_prune_calcu, inputs=[matrix_row_value, matrix_col_value, matrix_value_scale, unstructure_prune_calcu_method, unstructure_prune_calcu_amount], outputs=[calculate_result_content])

    return elem_dict