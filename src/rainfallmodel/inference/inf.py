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

from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
from ..common.parser import read_args
from ..common.env import get_default_device
from typing import TYPE_CHECKING, Any, Optional


def default_generate(user_conf:dict) -> None:
    """
    默认的输出逻辑
    """
    if 'model_path' not in user_conf:
        print("Oops!未配置model_path，无法根据路径加载模型，请确认配置")

    model_path = user_conf['model_path']

    device = get_default_device()
    
    # 第一步，加载Tokenizer并初始化生成配置，暂时先不支持可配置
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
    generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id or tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=20,
        do_sample=False,
    )

    # 第二步，循环生成
    while True:
        input_text = input(">>>")
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs["input_ids"], generation_config=generation_config)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("模型输出: ", result)


def do_infer(args: Optional[dict[str, Any]] = None):
    """
    模型推理使用
    """
    user_conf = read_args(args)
    default_generate(user_conf)
    



