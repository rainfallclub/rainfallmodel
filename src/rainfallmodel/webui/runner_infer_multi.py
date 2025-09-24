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

class InferMultiRunner(InferInterface):
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
        multi_base_model_path = get("infer.multi_base_model_path")
        multi_1_model_path = get("infer.multi_1_model_path")
        multi_2_model_path = get("infer.multi_2_model_path")
        multi_3_model_path = get("infer.multi_3_model_path")
        # 暂时不做重复性校验
        self.init_multi_backend(multi_base_model_path, multi_1_model_path, multi_2_model_path, multi_3_model_path)
        yield ALERTS["infer_backend_loaded"][lang]

    

    def unload_model(self) -> Generator:
        lang = self.manager.get_lang()
        yield ALERTS["infer_backend_unloading"][lang]
        self.baseline_backend = None
        self.comp1_backend = None
        self.comp2_backend = None
        self.comp3_backend = None
        torch_gc()
        yield ALERTS["infer_backend_unloaded"][lang]


    def interface_generate(
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

        # 至少需要加载一个
        # todo， 还是实现为哪种形式呢？
        if self.baseline_backend is None :
            lang = self.manager.get_lang()
            gr.Warning(ALERTS["infer_no_model"][lang])
            return chatbot, messages, ""
        
        """
        推理生成数据
        """  
        chatbot.append({"role": "user", "content": query})
        messages = messages + [{"role": "user", "content": query}]
        messages.append({"role": "user", "content": query})
        baseline_resp, comp1_resp,comp2_resp, comp3_resp = self.infer_multi_generate(query, max_new_tokens, top_p, temperature)
        resp = " 基准模型输出: \n" + baseline_resp + "\n  参考1输出: \n" + comp1_resp + "\n 参考2输出: \n" + comp2_resp + "\n 参考3输出: \n" + comp3_resp
        # print("产生的内容是:", resp)
        chatbot.append({"role": "assistant", "content": resp})
        return chatbot, messages, ""
        
        

