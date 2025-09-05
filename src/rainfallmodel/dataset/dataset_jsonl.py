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
from transformers import AutoTokenizer

def process_func(example: Dict[str, List], tokenizer: AutoTokenizer, dataset_conf:dict) -> dict:
    """
    使用分词器对数据进行处理
    """
    max_seq_length = int(dataset_conf['max_seq_length'])
    text_key = dataset_conf['text_field']

    encoded_texts = tokenizer(example[text_key], add_special_tokens=False)
    input_ids_list = encoded_texts['input_ids']

    new_input_ids_list, new_attn_mask_list = [], []
    for input_ids in input_ids_list:
        temp = input_ids[-max_seq_length+1:] + [tokenizer.eos_token_id]
        new_input_ids_list.append(temp)
        new_attn_mask_list.append([1] * len(temp))
    return {
        "input_ids": new_input_ids_list,
        "attention_mask": new_attn_mask_list
    }

def get_local_json_format_dataset(tokenizer:AutoTokenizer, dataset_conf:dict):
    """
    加载数据文件并使用指定的分词器进行处理
    """
    ds_train = load_dataset('json', data_files=dataset_conf['dataset_path'], split="train")
    ds_train = ds_train.map(
        lambda x: process_func(x, tokenizer, dataset_conf),
        batched=True,
        num_proc=1,
        remove_columns=ds_train.column_names
    )
    return ds_train



def get_hf_json_format_dataset(tokenizer:AutoTokenizer, dataset_conf:dict):
    """
    加载hugging face的数据文件并使用指定的分词器进行处理
    """
    ds_train = load_dataset(dataset_conf['dataset_path'], split='train')
    ds_train = ds_train.map(
        lambda x: process_func(x, tokenizer, dataset_conf),
        batched=True,
        num_proc=1,
        remove_columns=ds_train.column_names
    )
    return ds_train