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




from torch.utils.data import Dataset
from transformers import AutoTokenizer
from .dataset_pretrain_jsonl import get_pretrain_local_json_format_dataset, get_pretrain_hf_json_format_dataset
from .dataset_vocab_train_jsonl import (
    get_hf_json_format_dataset_for_vocab_train,
    get_local_json_format_dataset_for_vocab_train
)
from torch.utils.data import  random_split
from .dataset_conf import get_user_dataset_conf
from .dataset_path_handler import check_need_remote
from .dataset_sft_json import get_sft_local_json_format_dataset


def train_eval_split(dataset:Dataset, dataset_conf:dict) -> tuple[Dataset, Dataset]:
    """
    分割数据集
    """
    train_eval_percent = int(dataset_conf['train_eval_percent']) / 100
    train_size = int(train_eval_percent * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size]
    )
    return train_dataset, test_dataset


def get_pretrain_dataset(tokenizer:AutoTokenizer, user_conf:dict) -> tuple[Dataset, Dataset]:
    """
    获取预训练使用的数据集，目前暂时只支持jsonl格式
    目前暂时还不支持单独使用验证集，从训练集拆分出验证集
    这块后续会有较大的重构，暂时先这样实现
    """

    dataset_conf = get_user_dataset_conf(user_conf)

    # 判断是否需要远程连接
    need_remote, dataset_path = check_need_remote(dataset_conf['dataset_path'])
    dataset_conf['dataset_path'] = dataset_path
    dataset_conf['need_remote'] = need_remote

    if need_remote:
        dataset = get_pretrain_hf_json_format_dataset(tokenizer, dataset_conf)
    else:
        dataset = get_pretrain_local_json_format_dataset(tokenizer, dataset_conf)
    return train_eval_split(dataset, dataset_conf)





def get_vocab_train_dataset(user_conf:dict) -> tuple[Dataset, Dataset]:
    """
    获取词表训练的数据集，目前暂时只支持jsonl格式
    这块后续会有较大的重构，暂时先这样实现
    """

    dataset_conf = get_user_dataset_conf(user_conf)

    # 判断是否需要远程连接
    need_remote, dataset_path = check_need_remote(dataset_conf['dataset_path'])
    dataset_conf['dataset_path'] = dataset_path
    dataset_conf['need_remote'] = need_remote

    if need_remote:
        dataset = get_hf_json_format_dataset_for_vocab_train(dataset_conf)
    else:
        dataset = get_local_json_format_dataset_for_vocab_train(dataset_conf)
    # return train_eval_split(dataset, dataset_conf)
    return dataset




def get_sft_dataset(tokenizer:AutoTokenizer, user_conf:dict) -> tuple[Dataset, Dataset]:
    """
    获取微调使用的数据集，目前暂时只支持json格式
    这块后续会有较大的重构，暂时先这样实现
    """

    dataset_conf = get_user_dataset_conf(user_conf)

    # 判断是否需要远程连接,
    # 微调数据集暂不支持远程
    # 暂时不支持验证集
    # 判断是否需要远程连接
    need_remote, dataset_path = check_need_remote(dataset_conf['dataset_path'])
    dataset_conf['dataset_path'] = dataset_path
    dataset_conf['need_remote'] = need_remote
    
    return get_sft_local_json_format_dataset(tokenizer, dataset_conf)







