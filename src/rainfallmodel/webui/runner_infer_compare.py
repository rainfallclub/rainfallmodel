# Copyright 2025 the LlamaFactory team. and the RainFallModel team.
#
# This code is inspired by the LlamaFactory.
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/webui/runner.py
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



from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Optional
from ..common.packages import is_gradio_available
from .locales import ALERTS
from ..inference.infer_interface import InferInterface
from ..common.misc import torch_gc
    

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component
    from .manager import Manager

class InferCompareRunner(InferInterface):
    r"""A class to manage the running status."""

    def __init__(self, manager: "Manager") -> None:
        self.init_backend()
        self.manager = manager

    @property
    def loaded(self) -> bool:
        has_backend =  hasattr(self, 'backend')
        if not has_backend:
            return False
        return self.backend is not None

    def load_model(self, data) -> Generator:
        lang = self.manager.get_lang()
        yield ALERTS["infer_backend_loading"][lang]
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        teacher_model_path = get("infer.compare_teacher_model_path")
        student_model_path = get("infer.compare_student_model_path")
        # 暂时不做重复性校验
        self.init_compare_backend(teacher_model_path, student_model_path)
        yield ALERTS["infer_backend_loaded"][lang]

    

    def unload_model(self) -> Generator:
        lang = self.manager.get_lang()
        yield ALERTS["infer_backend_unloading"][lang]
        self.teacher_backend = None
        self.student_backend = None
        torch_gc()
        yield ALERTS["infer_backend_unloaded"][lang]


    def interface_chat(
        self, 
        chatbot: list[dict[str, str]], 
        messages: list[dict[str, str]], 
        query: str,
        max_new_tokens: int, 
        top_p: float, 
        temperature: float
        ) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
        """
        推理生成，暂时不支持流式输出，目前是一次性生成，后续再给出更完善的实现
        """

        # 如果没有推理后端，暂时实现为不报错，原数据返回
        # todo， 还是实现为哪种形式呢？
        if self.teacher_backend is None or self.student_backend is None:
            lang = self.manager.get_lang()
            gr.Warning(ALERTS["infer_no_model"][lang])
            return chatbot, messages, ""
        
        """
        推理生成数据
        """  
        chatbot.append({"role": "user", "content": query})
        messages = messages + [{"role": "user", "content": query}]
        messages.append({"role": "user", "content": query})
        teacher_resp, student_resp = self.infer_compare_chat(query, max_new_tokens, top_p, temperature)
        resp = " 教师模型输出: \n" + teacher_resp + "\n 学生模型输出: \n" + student_resp
        # print("产生的内容是:", resp)
        chatbot.append({"role": "assistant", "content": resp})
        return chatbot, messages, ""
        
        

    














