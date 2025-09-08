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


from datasets import load_dataset
from typing import Dict, List



def get_local_json_format_dataset_for_vocab_train(dataset_conf:dict) -> list:
    """
    加载本地数据文件
    """
    text_key = dataset_conf['text_field']
    ds_train = load_dataset('json', data_files=dataset_conf['dataset_path'], split="train")
    result = []
    for example in ds_train:
        result.append(example[text_key])
    return result



def get_hf_json_format_dataset_for_vocab_train(dataset_conf:dict):
    """
    加载hugging face的数据文件
    """
    text_key = dataset_conf['text_field']
    ds_train = load_dataset(dataset_conf['dataset_path'], split='train')
    result = []
    for example in ds_train:
        result.append(example[text_key])
    return result


