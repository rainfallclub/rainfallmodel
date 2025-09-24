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

    def init_multi_backend(self, baseline_model_path:str, comp1_path:str, comp2_path:str, comp3_path:str) -> None:
        self.baseline_backend = None
        self.comp1_backend = None
        self.comp2_backend = None
        self.comp3_backend = None
        if len(baseline_model_path) > 0:
            self.baseline_backend = BackendHf(baseline_model_path, None)
        if len(comp1_path) > 0 :
            self.comp1_backend = BackendHf(comp1_path, None)
        if len(comp2_path) > 0 :
            self.comp2_backend = BackendHf(comp2_path, None)
        if len(comp3_path) > 0:
            self.comp3_backend = BackendHf(comp3_path, None)
    
    
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
    
    def infer_multi_generate(self, query: str, max_new_tokens: int, top_p: float, temperature: float) -> str:
        """
        普通推理模式
        """
        baseline_resp = ""
        comp1_resp = ""
        comp2_resp = ""
        comp3_resp = ""
        if self.baseline_backend is not None:
            baseline_resp = self.baseline_backend.generate(query, max_new_tokens, top_p, temperature)
        if self.comp1_backend is not None:
            comp1_resp = self.comp1_backend.generate(query, max_new_tokens, top_p, temperature)
        if self.comp2_backend is not None:
            comp2_resp = self.comp2_backend.generate(query, max_new_tokens, top_p, temperature)
        if self.comp3_backend is not None:
            comp3_resp = self.comp3_backend.generate(query, max_new_tokens, top_p, temperature)
        return (baseline_resp, comp1_resp, comp2_resp, comp3_resp)

        




































