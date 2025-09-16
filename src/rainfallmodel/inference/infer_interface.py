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

from .backend_hf import BackendHf


class InferInterface:

    def __init__(self, model_path:str):
        """
        初始化后端推理，目前仅支持huggingface
        """
        print("model_path:  ", model_path)
        self.backend = BackendHf(model_path)

    
    
    def infer_generate(self, query: str, max_new_tokens: int, top_p: float, temperature: float) -> str:
        """
        普通推理模式
        """
        return self.backend.generate(query, max_new_tokens, top_p, temperature)
    

    def infer_chat(self, query: str, max_new_tokens: int, top_p: float, temperature: float) -> str:
        """
        聊天模式
        """
        return self.backend.chat(query, max_new_tokens, top_p, temperature)

        




































