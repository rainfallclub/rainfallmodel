# Copyright 2025 the LlamaFactory team. and the RainFallModel team.
#
# This code is inspired by the LlamaFactory.
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/webui/interface.py
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

import os
from ..common.packages import is_gradio_available
from .tab_pretrain import create_train_tab
from .tab_vocab_train import create_vocab_train_tab
from .tab_vocab_info import create_vocab_info_tab
from .tab_infer import create_infer_tab
from .tab_top import create_top
from .tab_todo import create_todo_tab
from .tab_sft import create_sft_tab
from .tab_distill import create_distill_tab
from .tab_export import create_format_tab
from .tab_prune import create_prune_tab
from .tab_quant import create_quant_tab
from .manager import Manager
from ..common.misc import is_env_enabled, fix_proxy

if is_gradio_available():
    import gradio as gr

def create_ui() -> "gr.Blocks":

    with gr.Blocks(title=f"RainFall Model") as demo:
        manager = Manager()
        
        gr.HTML("<h1><center>雨落大模型(RainFall Model): 简单且高效的从零训练大模型工具</center></h1>")
        gr.HTML(
            '<h3><center>代码地址: <a href="https://github.com/rainfallclub/rainfallmodel" target="_blank">点此看代码 </a>'
            '视频讲解: <a href=\"https://space.bilibili.com/3493279408588872\" target=\"_blank\"> 点此看视频</a>'
            # '使用文档: <a href=\"https://space.bilibili.com/3493279408588872\" target=\"_blank\"> 点此看文档</a>'
            '也可以在B站搜\"雨落实战\"这位UP主 </center></h3>'
        )
        # gr.HTML(
        #     '<h3><center>视频讲解: <a href="https://space.bilibili.com/3493279408588872" target="_blank">'
        #     "点此访问</a> 也可以在B站搜\"雨落实战\"这个UP主</center></h3>"
        # )
       
        manager.add_elems("top", create_top())
        with gr.Tab("模型预训练"):
            manager.add_elems("pretrain", create_train_tab(manager))
        with gr.Tab("词表构建"):
            manager.add_elems("vocab", create_vocab_train_tab(manager))
        with gr.Tab("词表检查"):
            manager.add_elems("vocab_info", create_vocab_info_tab(manager))
        with gr.Tab("模型推理"):
            manager.add_elems("infer", create_infer_tab(manager))
        with gr.Tab("模型微调"):
            manager.add_elems("sft", create_sft_tab(manager))
        with gr.Tab("模型蒸馏"):
            manager.add_elems("distill", create_distill_tab(manager))
        with gr.Tab("模型量化"):
            manager.add_elems("quant", create_quant_tab(manager))
        with gr.Tab("模型剪枝"):
            manager.add_elems("prune", create_prune_tab(manager))
        # with gr.Tab("数据集处理"):
        #     manager.add_elems("data", create_todo_tab(manager))
        with gr.Tab("模型格式与导出"):
            manager.add_elems("export", create_format_tab(manager))

    return demo


def run_web_ui() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    fix_proxy(ipv6_enabled=gradio_ipv6)
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)
