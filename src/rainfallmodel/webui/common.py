# Copyright 2025 the LlamaFactory team. and the RainFallModel team.
#
# This code is inspired by the LlamaFactory.
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/webui/common.py
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

from yaml import safe_dump
from typing import Any
import os


def save_cmd(args: dict[str, Any], output_dir, file_name) -> str:
    r"""Save CLI commands to launch training."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f:
        safe_dump(args, f)

    return os.path.join(output_dir, file_name)

def save_args(config_path: str, config_dict: dict[str, Any]) -> None:
    r"""Save the configuration to config path."""
    with open(config_path, "w", encoding="utf-8") as f:
        safe_dump(config_dict, f)
