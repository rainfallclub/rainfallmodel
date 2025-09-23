import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from ..model.model_manager import get_real_model_path
import gradio as gr

def do_merge_lora(base_model_path, lora_adapter_path, output_path):
    """
    将LoRA合并到基础模型并保存为独立模型

    Args:
        base_model_path (str): 原始基础模型的路径
        lora_adapter_path (str): LoRA 适配器权重所在的路径。
        output_path (str): 合并后模型的保存路径。
    """
    print("merge lora start...")

    # 1. 加载 tokenizer 和原始模型
    real_base_model_path = get_real_model_path(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(real_base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(real_base_model_path)

    # 2. 合并LoRA
    lora_model = PeftModel.from_pretrained(base_model,lora_adapter_path)
    merged_model = lora_model.merge_and_unload()

    # 3. 导出
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    gr.Info("LoRA合并完成")

    print("merge lora finished!")
