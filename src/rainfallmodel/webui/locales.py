# Copyright 2025 the LlamaFactory team. and the RainFallModel team.
#
# This code is inspired by the LlamaFactory.
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/webui/locales.py
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




ALERTS = {
    # 预训练相关信息
    "pretrain_dataset_not_exists": {
        "zh": "预训练时，数据集缺失，无法训练。",
    },

    # 推理相关信息
    "infer_backend_loading": {
        "zh": "模型状态: 模型正在玩命加载中，请耐心等待..."
    },
    "infer_backend_loaded": {
        "zh": "模型状态: 模型已加载"
    },
    "infer_backend_unloading": {
        "zh": "模型状态: 模型正在玩命卸载中，请耐心等待..."
    },
    "infer_backend_unloaded": {
        "zh": "模型状态: 模型已卸载"
    },
    "infer_backend_not_exist":{
        "zh": "后端模型缺失，无法生成文本，请先加载模型"
    },
    "infer_backend_unload":{
        "zh": "模型状态: 模型未加载"
    },

    # 词表构建相关信息
    "vocab_dataset_not_exists": {
        "zh": "构建词表时，数据集缺失，无法训练。",
    },
    "err_conflict": {
        "zh": "任务已存在，请先中断训练。",
    },
    "err_exists": {
        "zh": "模型已存在，请先卸载模型。",
    },
    "err_no_model": {
        "zh": "请选择模型。",
    },
    "err_no_path": {
        "zh": "模型未找到。",
    },

    # 微调时报错信息
    "sft_model_path_not_exist":{
        "zh": "模型未填写"
    },
    "sft_dataset_path_not_exist":{
        "zh": "数据集未填写"
    },
}












