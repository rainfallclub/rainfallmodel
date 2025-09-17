# Copyright 2025  the RainFallModel team.
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


from ..common.env import get_default_device

# 默认的训练配置，可以被用户配置进行覆盖
# 这个配置起到一个兜底的作用，避免值的缺失
default_sft_conf = {
    # 资源部分
    "model_path": "",
    "output_dir": "./output",
    "dataset_path": "", 

    #微调参数部分
    'ft_type': 'full', # full表示全量微调、freeze表示冻结后的全量微调、LoRA表示高效微调
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_target_modules": [], # 这里是一个列表，需要进行一下转换
    "lora_dropout": 0.0,

    # 数据加载部分
    "batch_size": 1, 
    "dataloader_num_workers": 1,

    # 计算部分
    "epochs": 3,
    "learning_rate":2e-4,
    "gradient_accumulation_steps": 4,
    "lr_scheduler_type":"cosine",
    
    # 保存部分
    "save_interval": 3,
    "save_total_limit": 3,
    "save_steps":1000,
    "save_safetensors":False,

    # 环境部分
    "device": "cpu", # 这里用cpu兜底，下面会进行判断

    # 监控部分
    "logging_steps":10,
    # swanlab上报
    "use_swanlab": True,
    "swanlab_project_name": "rainfall_sft_project",
    "swanlab_experiment_name": "rainfall_sft_experiment"
}





def get_user_sft_conf(user_conf:dict) -> dict:
    """
    获取用户定义的训练配置
    """
    default_sft_conf['device'] = get_default_device()
    default_sft_conf.update(user_conf)

    # 把用逗号分割的字符串转成列表
    lora_target_modules_str = default_sft_conf['lora_target_modules']
    default_sft_conf['lora_target_modules'] = comma_string_to_list(lora_target_modules_str)
    

    return default_sft_conf



def comma_string_to_list(input_string:str) -> list:
    """
    将逗号分隔的字符串转换为列表
    """
    items = input_string.split(",")
    items = [item.strip() for item in items]
    items = [item for item in items if item]
    return items

