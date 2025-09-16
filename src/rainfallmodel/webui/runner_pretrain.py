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


import os
from collections.abc import Generator
from copy import deepcopy
from subprocess import Popen, TimeoutExpired
from typing import TYPE_CHECKING, Any, Optional
from ..common.packages import is_gradio_available
from .locales import ALERTS
from .common import (
    save_args,
    save_cmd
)
    
from ..common.constants import RAINFALLBOARD_CONFIG, PRETRAIN_ARG_CONF_FILE_NAME

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component
    from .manager import Manager

class PretrainRunner:
    r"""A class to manage the running status."""

    def __init__(self, manager: "Manager") -> None:
        self.manager = manager
        self.trainer: Optional[Popen] = None

    def _validate_pretrain(self, data: dict["Component", Any]) -> str:
        r"""Validate the configuration."""
        lang = self.manager.get_lang()
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]

        dataset_path = get("pretrain.dataset_path")
        if not dataset_path:
            return ALERTS["pretrain_dataset_not_exists"][lang]

        return ""

    
    def _launch(self, data: dict["Component", Any]) -> Generator[dict["Component", Any], None, None]:
        r"""Start the process."""
        output_box = self.manager.get_elem_by_id("vocab.output_box")
        error = self._validate_pretrain(data)

        if error:
            gr.Warning(error)
            yield {output_box: error}
        else:
            # self.do_train, self.running_data = do_train, data
            args = self._parse_pretrain_args(data)
            output_dir = args["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            save_args(os.path.join(args["output_dir"], RAINFALLBOARD_CONFIG), self._build_config_dict(data))

            env = deepcopy(os.environ)
           
            # NOTE: DO NOT USE shell=True to avoid security risk
            self.trainer = Popen(["rainfallmodel", "pretrain", save_cmd(args, output_dir, PRETRAIN_ARG_CONF_FILE_NAME)], env=env)
            print("pretrain start.... please go to swanlab to check your work")
            # todo，实现监控功能
            # yield from self.monitor()

    def _parse_pretrain_args(self, data: dict["Component", Any]) -> dict[str, Any]:
        r"""Build and validate the training arguments."""
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        args = dict(
            tokenizer_path = get("pretrain.tokenizer_path"),
            output_dir = get("pretrain.output_dir"),
            dataset_path = get("pretrain.dataset_path"),
            text_field = get("pretrain.text_field"),
            max_seq_length = get("pretrain.max_seq_length"),
            train_eval_percent = get("pretrain.train_eval_percent"),
            # 模型参数相关
            tie_word_embeddings = get("pretrain.tie_word_embeddings"),
            use_cache = get("pretrain.use_cache"),
            hidden_size = get("pretrain.hidden_size"),
            max_position_embeddings = get("pretrain.max_position_embeddings"),
            num_hidden_layers = get("pretrain.num_hidden_layers"),
            initializer_range = get("pretrain.initializer_range"),
            num_attention_heads = get("pretrain.num_attention_heads"),
            num_key_value_heads = get("pretrain.num_key_value_heads"),
            attention_dropout = get("pretrain.attention_dropout"),
            attention_bias = get("pretrain.attention_bias"),
            intermediate_size = get("pretrain.intermediate_size"),
            mlp_bias = get("pretrain.mlp_bias"),
            hidden_act = get("pretrain.hidden_act"),
            rms_norm_eps = get("pretrain.rms_norm_eps"),
            rope_scaling = get("pretrain.rope_scaling"),
            rope_theta = get("pretrain.rope_theta"),

            # todo，暂时先注释，下个版本再删除
            # hidden_size = get("pretrain.hidden_size"),
            # intermediate_size = get("pretrain.intermediate_size"),
            # num_attention_heads = get("pretrain.num_attention_heads"),
            # num_hidden_layers = get("pretrain.num_hidden_layers"),
            # num_key_value_heads = get("pretrain.num_key_value_heads"),
            
            # 训练参数相关
            batch_size = get("pretrain.batch_size"),
            epochs = get("pretrain.epochs"),
            learning_rate = get("pretrain.learning_rate"),
            gradient_accumulation_steps = get("pretrain.gradient_accumulation_steps"),
            lr_scheduler_type = get("pretrain.lr_scheduler_type"),
            save_total_limit = get("pretrain.save_total_limit"),
            save_steps = get("pretrain.save_steps"),
            save_safetensors = get("pretrain.save_safetensors"),
            eval_strategy = get("pretrain.eval_strategy"),
            eval_steps = get("pretrain.eval_steps"),
            logging_steps = get("pretrain.logging_steps"),
            use_swanlab = get("pretrain.use_swanlab"),
            swanlab_project_name = get("pretrain.swanlab_project_name"),
            swanlab_experiment_name = get("pretrain.swanlab_experiment_name"),
        )
        return args
    
    def run_pretrain(self, data):
        yield from self._launch(data)

    def _build_config_dict(self, data: dict["Component", Any]) -> dict[str, Any]:
        r"""Build a dictionary containing the current training configuration."""
        config_dict = {}
        skip_ids = ["top.lang"]
        for elem, value in data.items():
            elem_id = self.manager.get_id_by_elem(elem)
            if elem_id not in skip_ids:
                config_dict[elem_id] = value

        return config_dict

    














