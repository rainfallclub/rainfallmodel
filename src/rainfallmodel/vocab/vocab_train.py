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



from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer
from tokenizers.trainers import Trainer
import os
import json
from typing import TYPE_CHECKING, Any, Optional
from ..common.parser import read_args
from ..dataset.dataset_manager import get_vocab_train_dataset
from .vocab_train_config import get_user_vocab_train_conf
from ..common.resource import get_root_path

def get_bpe_tokenizer_and_trainer(vocab_train_conf:dict) -> tuple[Tokenizer, Trainer]:
    """
    bpe算法，目前仅支持这一个
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    vocab_size = int(vocab_train_conf['vocab_size'])
    min_frequency = int(vocab_train_conf['min_frequency'])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens = get_special_tokens(vocab_train_conf),
        show_progress=True,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    )

    return (tokenizer, trainer)

def get_unigram_tokenizer_and_trainer(vocab_train_conf:dict) -> tuple[Tokenizer, Trainer]:
    """
    unigram算法，后续再支持
    """
    pass

def get_special_tokens(vocab_train_conf:dict) -> list:
    """
    获取特殊token
    """
    result = set()
    # vocab_train_conf['special_tokens']默认是列表，需要转换为集合
    result = result.union(set(vocab_train_conf['special_tokens'])) 
    result.add(vocab_train_conf['bos_token'])
    result.add(vocab_train_conf['eos_token'])
    result.add(vocab_train_conf['padding_token'])
    result.add(vocab_train_conf['unknown_token'])
    return list(result)




def get_tokenizer_config(vocab_train_conf:dict, tokenizer:Tokenizer):
    """
    获取tokenizer_config.json文件所需的内容
    """
    default_config_json = get_root_path() + "/data/internal/basic_vocab_config.json"
    with open(default_config_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data['bos_token'] = vocab_train_conf['bos_token']
    data['eos_token'] = vocab_train_conf['eos_token']
    data['pad_token'] = vocab_train_conf['padding_token']
    data['unk_token'] = vocab_train_conf['unknown_token']
    tokens_decoder = dict()
    for spec_token in  get_special_tokens(vocab_train_conf):
        key = tokenizer.encode(spec_token).ids[0]
        str_key = str(key)
        spec_token_val = dict(
            content=spec_token,
            lstrip=False,
            normalized=False,
            rstrip=False,
            single_word=False,
            special=True
        )
        tokens_decoder[str_key] = spec_token_val
    data['added_tokens_decoder'] = tokens_decoder
    return data



def do_vocab_train(args: Optional[dict[str, Any]] = None) -> None:
    """
    词表训练的主流程，目前先这样实现
    """

    # 第一步，读取配置并进行转换
    user_conf = read_args(args)
    vocab_train_conf = get_user_vocab_train_conf(user_conf)

    # 第二步，获取tokenizer和trainer
    print("vocab train start...")
    tokenizer, trainer = get_bpe_tokenizer_and_trainer(vocab_train_conf)

    # 第三步，加载数据并训练
    dataset = get_vocab_train_dataset(vocab_train_conf)
    tokenizer.train_from_iterator(dataset, trainer)
    tokenizer.decoder = decoders.ByteLevel()

    # 第四步，执行保存逻辑
    output_dir = vocab_train_conf['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    # tokenizer.model.save(output_dir)  # 感兴趣的话可以查看下这个是什么

    # 第五步，保存tokenizer配置
    config = get_tokenizer_config(vocab_train_conf, tokenizer)
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("vocab  train success!!")


















