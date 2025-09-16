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

if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

def create_distill_tab(manager: "Manager") -> dict[str, "Component"]:
    input_elems = manager.get_base_elems()
    elem_dict = dict()


    with gr.Blocks():
        with gr.Tab("黑盒蒸馏"):
            hidden_size_list =[256, 768, 1024, 4096]
            hidden_size = gr.Dropdown(choices=hidden_size_list, label="全量微调配置1", value=256, interactive=True, allow_custom_value=True)
            intermediate_size_list =[768, 3072, 4096, 11008]
            intermediate_size = gr.Dropdown(choices=intermediate_size_list, label="全量微调配置2", value=768, interactive=True, allow_custom_value=True)
        with gr.Tab("白盒蒸馏"):
            hidden_size_list =[256, 768, 1024, 4096]
            hidden_size = gr.Dropdown(choices=hidden_size_list, label="全量微调配置1", value=256, interactive=True, allow_custom_value=True)
            intermediate_size_list =[768, 3072, 4096, 11008]
            intermediate_size = gr.Dropdown(choices=intermediate_size_list, label="全量微调配置2", value=768, interactive=True, allow_custom_value=True)
       

    gr.Markdown("---") 
    gr.Markdown("#### 资源配置")
    with gr.Row():
        dataset_path_list = get_dataset_config()
        dataset_path = gr.Dropdown(choices=dataset_path_list, label="数据集文件(支持本地文件，也可以下拉框中选择)",  interactive=True, allow_custom_value=True)

    with gr.Row():
        model_path_list = get_model_config()
        model_path = gr.Dropdown(choices=model_path_list, label="模型路径或地址",  interactive=True, allow_custom_value=True)
    
    with gr.Row():
        output_path_list = [get_output_path() + "_sft"]
        output_dir = gr.Dropdown(choices=output_path_list, label="输出路径",  interactive=True, allow_custom_value=True)

    input_elems.update({dataset_path, output_dir, model_path})
    elem_dict.update(
        dict(
            dataset_path=dataset_path, 
            output_dir=output_dir, 
            model_path=model_path
        )
    )

    gr.Markdown("---") 
    gr.Markdown("#### 微调类型配置")
    with gr.Blocks():
        with gr.Tab("全量微调"):
            hidden_size_list =[256, 768, 1024, 4096]
            hidden_size = gr.Dropdown(choices=hidden_size_list, label="全量微调配置1", value=256, interactive=True, allow_custom_value=True)
            intermediate_size_list =[768, 3072, 4096, 11008]
            intermediate_size = gr.Dropdown(choices=intermediate_size_list, label="全量微调配置2", value=768, interactive=True, allow_custom_value=True)
        with gr.Tab("freeze微调"):
            hidden_size_list =[256, 768, 1024, 4096]
            hidden_size = gr.Dropdown(choices=hidden_size_list, label="freeze微调配置1", value=256, interactive=True, allow_custom_value=True)
            intermediate_size_list =[768, 3072, 4096, 11008]
            intermediate_size = gr.Dropdown(choices=intermediate_size_list, label="freeze微调配置2", value=768, interactive=True, allow_custom_value=True)
        with gr.Tab("LoRA微调"):
            hidden_size_list =[256, 768, 1024, 4096]
            hidden_size = gr.Dropdown(choices=hidden_size_list, label="LoRA微调配置1", value=256, interactive=True, allow_custom_value=True)
            intermediate_size_list =[768, 3072, 4096, 11008]
            intermediate_size = gr.Dropdown(choices=intermediate_size_list, label="LoRA微调配置2", value=768, interactive=True, allow_custom_value=True)  

    gr.Markdown("---") 
    gr.Markdown("#### 数据加载与计算部分(请参考机器配置)")
    with gr.Row():
        batch_size = gr.Textbox(label="训练批次batch_size", value=4, interactive=True)
        epochs = gr.Textbox(label="训练轮次epoch", value=3, interactive=True)
        learning_rate = gr.Textbox(label="学习率learning_rate", value='2e-4', interactive=True)
        gradient_accumulation_steps = gr.Textbox(label="梯度累积步数", value=4, interactive=True)
        lr_scheduler_type = gr.Textbox(label="学习率调整策略lr_scheduler_type", value='cosine', interactive=True)
    
    input_elems.update({batch_size, epochs, learning_rate, gradient_accumulation_steps, lr_scheduler_type})
    elem_dict.update(
        dict(
            batch_size=batch_size, 
            epochs=epochs, 
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type=lr_scheduler_type
        )
    )    

    gr.Markdown("---") 
    gr.Markdown("#### 模型保存部分")
    with gr.Row():
        save_total_limit = gr.Textbox(label="共计保存限制", value=5, interactive=True)
        save_steps = gr.Textbox(label="每多少步保存一次", value=1000, interactive=True)
        save_safetensors_list = [True, False]
        save_safetensors = gr.Dropdown(choices=save_safetensors_list, label="是否使用SafeTensor格式", value=False, interactive=True)
       
    input_elems.update({save_total_limit, save_steps, save_safetensors})
    elem_dict.update(
        dict(
            save_total_limit=save_total_limit, 
            save_steps=save_steps, 
            save_safetensors=save_safetensors
        )
    )  


    gr.Markdown("---") 
    gr.Markdown("#### 其他配置")
    with gr.Blocks():
        with gr.Tab("Eval配置"):
             with gr.Row():
                eval_strategy_value_list = ["epoch", "steps", "no"]
                eval_strategy = gr.Dropdown(choices=eval_strategy_value_list, label="验证集策略", value="steps", interactive=True, allow_custom_value=True)
                eval_steps = gr.Textbox(label="每多少步验证一次(仅设置按步数验证时有效)", value=1000, interactive=True)
                input_elems.update({eval_strategy, eval_steps})
                elem_dict.update(
                    dict(
                        eval_strategy=eval_strategy, 
                        eval_steps=eval_steps
                    )
                )
        with gr.Tab("日志&监控配置"):
             with gr.Row():
                logging_steps = gr.Textbox(label="每多少步记录一次日志", value=10, interactive=True)
                use_swanlab_list = [True, False]
                use_swanlab = gr.Dropdown(choices=use_swanlab_list, label="是否使用Swanlab", value=True, interactive=True)
                swanlab_project_name = gr.Textbox(label="swanlab项目名", value="rainfall_project", interactive=True)
                swanlab_experiment_name = gr.Textbox(label="swanlab实验名", value="rainfall_experiment", interactive=True)
                input_elems.update({logging_steps, use_swanlab, swanlab_project_name,swanlab_experiment_name})
                elem_dict.update(
                    dict(
                        logging_steps=logging_steps, 
                        use_swanlab=use_swanlab,
                        swanlab_project_name=swanlab_project_name,
                        swanlab_experiment_name=swanlab_experiment_name
                    )
                )
 
        
    with gr.Row():
        start_btn = gr.Button(value="开始训练", variant="primary")
       

    # 暂时先关掉，后面再实现
    with gr.Column(scale=1):
        loss_viewer = gr.Plot(visible=False)
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
    output_elems = [output_box, progress_bar, loss_viewer]
    # start_btn.click(manager.pretrain_runner.run_pretrain, input_elems, output_elems)


    return elem_dict


    