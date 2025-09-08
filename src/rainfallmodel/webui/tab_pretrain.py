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
from ..common.resource import get_dataset_config, get_tokenizer_config, get_output_path

if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

def create_train_tab(manager: "Manager") -> dict[str, "Component"]:
    input_elems = manager.get_base_elems()
    elem_dict = dict()

    gr.Markdown("---") 
    gr.Markdown("#### 资源配置")
    with gr.Row():
        dataset_path_list = get_dataset_config()
        dataset_path = gr.Dropdown(choices=dataset_path_list, label="数据集文件(支持本地文件，也可以下拉框中选择)",  interactive=True, allow_custom_value=True)

    with gr.Row():
        tokenizer_path_list = get_tokenizer_config()
        tokenizer_path = gr.Dropdown(choices=tokenizer_path_list, label="词表文件",  interactive=True, allow_custom_value=True)

    with gr.Row():
        output_path_list = [get_output_path() + "_pretrain"]
        output_dir = gr.Dropdown(choices=output_path_list, label="输出路径",  interactive=True, allow_custom_value=True)

    input_elems.update({dataset_path, output_dir, tokenizer_path})
    elem_dict.update(
        dict(
            dataset_path=dataset_path, 
            output_dir=output_dir, 
            tokenizer_path=tokenizer_path
        )
    )

    gr.Markdown("---") 
    gr.Markdown("#### 数据集相关配置")
    with gr.Row():
        text_field = gr.Textbox(label="jsonl格式的文本key", value="text", interactive=True)
        max_seq_length = gr.Textbox(label="最大长度", value=1024,  interactive=True)
        train_eval_percent = gr.Textbox(label="无单独验证集时训练内容占比", value=99, interactive=True)

    input_elems.update({text_field, max_seq_length, train_eval_percent})
    elem_dict.update(
        dict(
            text_field=text_field, 
            max_seq_length=max_seq_length, 
            train_eval_percent=train_eval_percent
        )
    )



    gr.Markdown("---") 
    gr.Markdown("#### 模型结构配置")
    with gr.Row():
        hidden_size_list =[256, 768, 1024, 4096]
        hidden_size = gr.Dropdown(choices=hidden_size_list, label="隐藏层大小", value=256, interactive=True, allow_custom_value=True)
        intermediate_size_list =[768, 3072, 4096, 11008]
        intermediate_size = gr.Dropdown(choices=intermediate_size_list, label="中间层大小", value=768, interactive=True, allow_custom_value=True)
        num_attention_heads_list =[12, 16, 32, 40]
        num_attention_heads = gr.Dropdown(choices=num_attention_heads_list, label="注意力头数", value=16, interactive=True, allow_custom_value=True)
        num_hidden_layers_list =[4, 12, 24, 32]
        num_hidden_layers = gr.Dropdown(choices=num_hidden_layers_list, label="Transformer块层数", value=4, interactive=True, allow_custom_value=True)
        num_key_value_heads_list =[2, 8]
        num_key_value_heads = gr.Dropdown(choices=num_key_value_heads_list, label="kv头数", value=8, interactive=True, allow_custom_value=True)
    
    input_elems.update({hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, num_key_value_heads})
    elem_dict.update(
        dict(
            hidden_size=hidden_size, 
            intermediate_size=intermediate_size, 
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads
        )
    )    

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
    gr.Markdown("#### 验证集eval部分")
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

    gr.Markdown("---") 
    gr.Markdown("#### 日志&监控部分")
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
    start_btn.click(manager.pretrain_runner.run_pretrain, input_elems, output_elems)


    return elem_dict


    