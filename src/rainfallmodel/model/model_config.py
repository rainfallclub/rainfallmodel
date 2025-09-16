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

import json
import re
import torch
from transformers import  AutoTokenizer
from ..common.env import get_default_device


# 模型的默认配置
# 如果用户有特定的配置，则会使用用户特定的配置进行覆盖

default_model_conf = {

    # 整体架构
    "tie_word_embeddings": True,
    "use_cache": True,
    "hidden_size":256,
    "max_position_embeddings": 2048,
    "num_hidden_layers":4,
    "initializer_range":0.02,
    
    # 注意力相关
    "num_attention_heads":16,
    "num_key_value_heads":8,
    "attention_dropout": 0.0,
    "attention_bias": False,

    # moe/ffn/swiglu相关
    "intermediate_size":768,
    "mlp_bias": False,
    "hidden_act": "silu",

    #词表相关【从tokenizer生成，但是也给了默认值】
    "vocab_size": 6400,
    "bos_token_id": 1,
    "eos_token_id": 2,

    # 归一化配置
    "rms_norm_eps": 1e-6,

    # rope相关配置
    "rope_scaling": None,
    "rope_theta": 10000.0,

    # 环境参数
    'dtype': torch.float32,
    'device':'cpu', # 使用cpu作为兜底，会在特定位置进行处理

}

def get_valid_rope_scaling_string(rope_scaling_str:str):
    """
    获取rope_scaling一个合法的取值，如果不是合法配置，则使用None，
    这样是否有点暴力了呢？
    """
    if rope_scaling_str is None or not rope_scaling_str.strip():
        return True, None

    try:
        config = json.loads(rope_scaling_str)
        if not isinstance(config, dict):
            return False, "必须是JSON对象格式，而不是数组或基本类型"
        
        if "type" not in config:
            return False, "缺少必需的type字段"
        
        if "factor" not in config:
            return False, "缺少必需的factor字段"

        return True, config
    except json.JSONDecodeError:
        return True, None

def get_user_model_config(user_conf:dict, tokenizer:AutoTokenizer) -> dict:
    """
    获取用户配置过之后的模型配置
    这里直接使用用户配置进行覆盖，不做有效性检测
    """
    
    # 设备部分处理
    default_model_conf['device'] = get_default_device() 
    default_model_conf.update(user_conf)

    # 词表部分处理
    default_model_conf['vocab_size'] = tokenizer.vocab_size
    default_model_conf['eos_token_id'] = tokenizer.eos_token_id
    default_model_conf['bos_token_id'] = tokenizer.bos_token_id

    # rope_scaling 特殊处理一下，避免传入非法值,传入的是字符串，需要的是None或者字典类型
    rope_scaling = default_model_conf['rope_scaling']
    rope_scaling_flag, rope_scaling_config = get_valid_rope_scaling_string(rope_scaling)
    if not rope_scaling_flag:
        print('rope_scaling config error, ignore!')
        default_model_conf['rope_scaling'] = None
    else:
        default_model_conf['rope_scaling'] = rope_scaling_config

    return default_model_conf


















