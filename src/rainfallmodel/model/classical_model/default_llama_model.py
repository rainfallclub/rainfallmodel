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

from transformers import  AutoModelForCausalLM, AutoConfig


def get_llama_model(model_conf:dict) -> AutoModelForCausalLM:
    """
    默认使用Llama架构，可满足大多数模型练手的需求,
    如果有特殊定制化诉求，可以在custom_model中实现

    
    参数:
    model_conf: 模型配置
    """
    
    config = AutoConfig.for_model(
        model_type="llama",
        hidden_size=int(model_conf['hidden_size']),
        intermediate_size=int(model_conf['intermediate_size']),
        num_attention_heads=int(model_conf['num_attention_heads']),
        num_hidden_layers=int(model_conf['num_hidden_layers']),
        num_key_value_heads=int(model_conf['num_key_value_heads'])
    )
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=model_conf['dtype']
    ).to(model_conf['device'])

    
    return model