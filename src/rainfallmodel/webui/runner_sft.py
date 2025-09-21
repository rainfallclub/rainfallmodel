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
    
from ..common.constants import RAINFALLBOARD_CONFIG, SFT_ARG_CONF_FILE_NAME

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component
    from .manager import Manager

class SftRunner:
    r"""A class to manage the running status."""

    def __init__(self, manager: "Manager") -> None:
        self.manager = manager
        self.trainer: Optional[Popen] = None
        self.do_train = True
        self.running_data: dict[Component, Any] = None
        self.aborted = False
        self.running = False

    def _validate_param(self, data: dict["Component", Any]) -> str:
        r"""Validate the configuration."""
        lang = self.manager.get_lang()
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]

        model_path = get("sft.model_path")
        if not model_path:
            return ALERTS["sft_model_path_not_exist"][lang]

        dataset_path = get("sft.dataset_path")
        if not dataset_path:
            return ALERTS["sft_dataset_path_not_exist"][lang]

        return ""

    
    def _launch(self, data: dict["Component", Any]) -> Generator[dict["Component", Any], None, None]:
        r"""Start the process."""
        output_box = self.manager.get_elem_by_id("sft.output_box")
        error = self._validate_param(data)

        if error:
            gr.Warning(error)
            yield {output_box: error}
        else:
            gr.Info(ALERTS["sft_doing"][self.manager.get_lang()])
            args = self._parse_sft_args(data)
            output_dir = args["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            save_args(os.path.join(args["output_dir"], RAINFALLBOARD_CONFIG), self._build_config_dict(data))

            env = deepcopy(os.environ)

            # NOTE: DO NOT USE shell=True to avoid security risk
            self.trainer = Popen(["rainfallmodel", "sft", save_cmd(args, output_dir, SFT_ARG_CONF_FILE_NAME)], env=env)
            # todo，实现监控功能
            # yield from self.monitor()

    def _parse_sft_args(self, data: dict["Component", Any]) -> dict[str, Any]:
        r"""Build and validate the sft arguments."""
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        args = dict(
            model_path = get("sft.model_path"),
            dataset_path = get("sft.dataset_path"),
            output_dir = get("sft.output_dir"),
            ft_type = get("sft.ft_type"),
            lora_rank = get("sft.lora_rank"),
            lora_alpha = get("sft.lora_alpha"),
            lora_target_modules = get("sft.lora_target_modules"),
            lora_dropout=get("sft.lora_dropout"),
            batch_size = get("sft.batch_size"),
            epochs = get("sft.epochs"),
            learning_rate = get("sft.learning_rate"),
            gradient_accumulation_steps = get("sft.gradient_accumulation_steps"),
            lr_scheduler_type = get("sft.lr_scheduler_type"),
            save_total_limit = get("sft.save_total_limit"),
            logging_steps = get("sft.logging_steps"),
            use_swanlab = get("sft.use_swanlab"),
            swanlab_project_name = get("sft.swanlab_project_name"),
            swanlab_experiment_name = get("sft.swanlab_experiment_name"),
        )
        return args
    
    def run_sft_train(self, data):
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

    














