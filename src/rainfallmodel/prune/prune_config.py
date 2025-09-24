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
default_pretrain_conf = {
    # 资源部分
    "model_path": "",
    "output_dir": "./output",
   

    # 数据加载部分


}





def get_user_prune_conf(user_conf:dict) -> dict:
    """
    获取用户定义的训练配置
    """
    default_pretrain_conf['device'] = get_default_device()
    default_pretrain_conf.update(user_conf)
    return default_pretrain_conf


def comma_string_to_list(input_string:str) -> list:
    """
    将逗号分隔的字符串转换为列表
    """
    items = input_string.split(",")
    items = [item.strip() for item in items]
    items = [item for item in items if item]
    return items
















