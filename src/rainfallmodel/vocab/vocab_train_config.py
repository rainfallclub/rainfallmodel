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


default_vacab_train_conf = {

    # 资源部分
    "output_dir": "./output",
    "dataset_path": "", 

    #计算部分
    "vocab_size": 6400,
    "min_frequency": 2,

    # 特殊token部分
    "padding_token": "<|im_end|>",
    "unknown_token": "<|im_end|>",
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "special_tokens": []
    
}


def get_user_vocab_train_conf(user_conf:dict) -> dict:
    """
    获取用户定义的训练配置
    """

    # 对于special_tokens这个要做特殊处理，用空格拆分为列表
    special_tokens = user_conf['special_tokens']
    if special_tokens is not None:
        user_conf['special_tokens'] = special_tokens.split()
    else:
        user_conf["special_tokens"] = []
    

    default_vacab_train_conf.update(user_conf)
    return default_vacab_train_conf














