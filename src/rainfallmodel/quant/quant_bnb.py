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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from ..model.model_manager import get_real_model_path
import gradio as gr

def get_8bit_quant_config() -> BitsAndBytesConfig:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
    )
    return quantization_config

def get_4bit_quant_config() -> BitsAndBytesConfig:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  
        bnb_4bit_use_double_quant=True,  
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16,  
        
        # llm_int8_enable_fp32_cpu_offload=True,
        # llm_int8_threshold=6.0,
        # llm_int8_skip_modules=None,
    )
    return quantization_config

def do_quant_bnb(quant_conf:dict) -> None:
    """
    基于bnb的量化，后续再支持更多的情况
    """
    print("quant start...")

    # 第一步，加载模型
    model_path = quant_conf['model_path']
    real_model_path = get_real_model_path(model_path)


    # 第二步，选择量化配置
    quant_type = quant_conf['quant_type']
    if '8bit' == quant_type:
        quantization_config = get_8bit_quant_config()
    elif '4bit' == quant_type:
        quantization_config = get_4bit_quant_config()
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        real_model_path,
        quantization_config=quantization_config,
        device_map="auto",  
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(real_model_path)

    # 第二步，执行保存
    output_dir = quant_conf['output_dir']
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("quant success!")



def do_quant_bnb_interface(model_path:str, ouput_dir:str, quant_type:str) -> None:
    quant_conf = dict(
        model_path = model_path,
        output_dir = ouput_dir,
        quant_type = quant_type
    )

    do_quant_bnb(quant_conf)
    gr.Info("量化操作已完成!")












