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


from pathlib import Path
import json
from datetime import datetime

def get_root_path() -> str:
    """
    获取RainFallModel的根路径，这个是绝对路径
    """
    root_path = Path(__file__).parents[3].absolute()
    return str(root_path)

def get_data_config() -> dict:
    """
    获取data_conf.json文件的相关内容
    """
    root_path = get_root_path()
    data_conf_path = root_path + "/data/data_conf.json"
    data = {}
    with open(data_conf_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_dataset_config() -> dict:
    """
    获取内置数据集的配置
    """
    data = get_data_config()
    return data["dataset"]

def get_tokenizer_config() -> dict:
    """
    获取内置的词表的配置
    """
    data = get_data_config()
    return data["tokenizer"]

def get_model_config() -> dict:
    """
    获取内置数据集的配置
    """
    data = get_data_config()
    return data["model"]


def get_output_path() -> str:
    """
    获取输出目录，自动生成，根据当前时间设置
    """
    root_path = get_root_path()
    now = datetime.now()
    time_suffix = now.strftime("%Y_%m_%d_%H_%M_%S")
    return root_path + "/output/" + time_suffix






















