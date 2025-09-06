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
from ..common.env import get_default_device


class BackendHf:
    """
    基于HuggingFace的transformers、tokenizers库实现的推理后端
    目前只有这一个实现，后续会支持更多实现
    """

    def __init__(self, model_path: str):
        device = get_default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()

    def generate(self, query: str, max_new_tokens: int, top_p: float, temperature: float) -> str:
        generation_config = GenerationConfig(
            bos_token_id=self.tokenizer.bos_token_id or self.tokenizer.cls_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p = top_p,
            temperature = temperature
        )        
        inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs["input_ids"], generation_config=generation_config)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
        


