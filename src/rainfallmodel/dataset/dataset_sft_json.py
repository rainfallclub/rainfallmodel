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

from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset


def process_func(example, tokenizer, dataset_conf:dict) -> dict:
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )

    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    max_seq_length = int(dataset_conf['max_seq_length'])
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   



def get_sft_local_json_format_dataset(tokenizer:AutoTokenizer, dataset_conf:dict) -> Dataset:
    """
    获取用于sft的本地json格式的数据集
    """
    dataset_path = dataset_conf['dataset_path']
    train_df = pd.read_json(dataset_path)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map( 
        lambda x: process_func(x, tokenizer, dataset_conf),
        remove_columns=train_ds.column_names)
    return train_dataset