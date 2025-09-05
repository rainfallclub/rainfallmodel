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
from ..common.resource import get_tokenizer_config, get_root_path


def get_vocab_size(tokenizer:AutoTokenizer) -> int:
    return tokenizer.vocab_size

def get_tokenizer(tokenizer_name:str) -> AutoTokenizer:
    vocab = get_tokenizer_config()
    if tokenizer_name not in vocab:
        return None
    
    tokenizer_item = vocab[tokenizer_name]
    if tokenizer_item["type"] == "local_file_relative_path":
        file_path = tokenizer_item["file_path"]
        tokenizer_path = get_root_path() + "/" + file_path
    elif tokenizer_item["type"] == "hugging_face_repo":
        tokenizer_path = tokenizer_item["repo"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def get_tokenizer_info() -> dict:
    pass


def tokenizer_encode(str_input:str, tokenizer: AutoTokenizer) -> list:
    return tokenizer.encode(str_input)



def tokenizer_decode(input_ids:str, tokenizer: AutoTokenizer) -> str:
    return tokenizer.decode(input_ids)
