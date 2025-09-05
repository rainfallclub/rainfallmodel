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

def get_bpe_tokenizer_and_trainer(args: Optional[dict[str, Any]] = None) -> tuple[Tokenizer, Trainer]:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    vocab_size = int(args['vocab_size'])
    min_frequency = int(args['vocab_min_frequency'])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens = get_special_tokens(args),
        show_progress=True,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    )

    return (tokenizer, trainer)

def get_unigram_tokenizer_and_trainer(args: Optional[dict[str, Any]] = None) -> tuple[Tokenizer, Trainer]:
    pass

def get_special_tokens(args: Optional[dict[str, Any]] = None) -> list:
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    return special_tokens

def vocab_train(args: Optional[dict[str, Any]] = None) -> None:
    args = read_args(args)
    tokenizer, trainer = get_bpe_tokenizer_and_trainer(args)
    # 定义特殊tokens
  

    dataset_path = args['vocab_dataset_path']
    output_dir = args['vocab_output_dir']

    texts = read_data(dataset_path)
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.decoder = decoders.ByteLevel()
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    tokenizer.model.save(output_dir)

    print("vocab  train success!")


# 读取数据
def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']  















