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

from transformers import  PreTrainedModel, AutoTokenizer
from .classical_model.default_llama_model import get_llama_model
from .model_config import get_user_model_config
from ..common.resource import get_model_config


def get_pretrain_model(user_conf:dict, tokenizer:AutoTokenizer) -> PreTrainedModel:
    """
    暂时仅支持llama，后续会完善更加复杂的需求
    """

    model_conf = get_user_model_config(user_conf, tokenizer)
    return get_llama_model(model_conf)


def get_real_model_path(model_path: str) -> str:
    """
    根据model_path判断是否需要远程拉取
    获取真正的模型路径地址信息
    在微调、蒸馏等流程中会使用到
    """
    model_dict = get_model_config()
    if model_path not in model_dict:
        return model_path
    
    model_conf = model_dict[model_path]
    return model_conf['repo']


