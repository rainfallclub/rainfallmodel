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

        tie_word_embeddings=bool(model_conf['tie_word_embeddings']),
        use_cache=bool(model_conf['use_cache']),
        hidden_size=int(model_conf['hidden_size']),
        max_position_embeddings=int(model_conf['max_position_embeddings']),
        num_hidden_layers=int(model_conf['num_hidden_layers']),
        initializer_range=float(model_conf['initializer_range']),
        
        
        num_attention_heads=int(model_conf['num_attention_heads']),
        num_key_value_heads=int(model_conf['num_key_value_heads']),
        attention_dropout=float(model_conf['attention_dropout']),
        attention_bias=bool(model_conf['attention_bias']),


        intermediate_size=int(model_conf['intermediate_size']),
        mlp_bias=bool(model_conf['mlp_bias']),
        hidden_act=model_conf['hidden_act'],


        vocab_size=int(model_conf['vocab_size']),
        bos_token_id=int(model_conf['bos_token_id']),
        eos_token_id=int(model_conf['eos_token_id']),

        rms_norm_eps=float(model_conf['rms_norm_eps']),

        rope_scaling=model_conf['rope_scaling'],
        rope_theta=float(model_conf['rope_theta']),
        
    )
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=model_conf['dtype']
    ).to(model_conf['device'])

    
    return model