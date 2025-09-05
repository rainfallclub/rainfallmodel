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



# 默认的数据集配置
default_dataset_conf = {

    'text_field':'text',
    'max_seq_length':1024,
    'dataset_path':'',
    'train_eval_percent':99,  # 训练集在整体数据集中的占比

    #
    
}


def get_user_dataset_conf(user_conf:dict) -> dict:
    """
    获取用户定义的数据集配置
    """
    default_dataset_conf.update(user_conf)
    return default_dataset_conf


