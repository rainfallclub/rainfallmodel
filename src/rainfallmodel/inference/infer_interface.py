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

    def init_backend(self):
        """
        初始化后端推理，目前仅支持huggingface生态
        """
        # print("model_path:  ", model_path)
        self.base_backend = None
        self.chat_backend = None
        self.teacher_backend = None
        self.student_backend = None

    def init_base_backend(self, model_path:str) -> None:
        self.base_backend = BackendHf(model_path, None)

    def init_chat_backend(self, model_path:str, lora_path:str) -> None:
        self.chat_backend = BackendHf(model_path, lora_path)

    def init_compare_backend(self, teacher_model_path:str, student_model_path:str) -> None:
        self.teacher_backend = BackendHf(teacher_model_path, None)
        self.student_backend = BackendHf(student_model_path, None)
    
    
    def infer_generate(self, query: str, max_new_tokens: int, top_p: float, temperature: float) -> str:
        """
        普通推理模式
        """
        return self.base_backend.generate(query, max_new_tokens, top_p, temperature)
    

    def infer_chat(self, query: str, max_new_tokens: int, top_p: float, temperature: float) -> str:
        """
        聊天模式
        """
        return self.chat_backend.chat(query, max_new_tokens, top_p, temperature)
    
    def infer_compare_chat(self, query: str, max_new_tokens: int, top_p: float, temperature: float) -> str:
        """
        对比模式
        """
        teacher_resp =  self.teacher_backend.chat(query, max_new_tokens, top_p, temperature)
        student_resp = self.student_backend.chat(query, max_new_tokens, top_p, temperature)
        return (teacher_resp, student_resp)

        




































