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


import torch
from ..common.env import get_default_device


# 模型的默认配置
# 如果用户有特定的配置，则会使用用户特定的配置进行覆盖

default_model_conf = {

    # 模型架构
    "hidden_size":256,
    "intermediate_size":768,
    "num_attention_heads":16,
    "num_hidden_layers":4,
    "num_key_value_heads":8,


    # 环境参数
    'dtype': torch.float32,
    'device':'cpu', # 使用cpu作为兜底，会在特定位置进行处理

}


def get_user_model_config(user_conf:dict) -> dict:
    """
    获取用户配置过之后的模型配置
    这里直接使用用户配置进行覆盖，不做有效性检测
    """

    default_model_conf['device'] = get_default_device() 
    default_model_conf.update(user_conf)
    return default_model_conf


















