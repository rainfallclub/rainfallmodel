# Copyright 2025 the LlamaFactory team. and the RainFallModel team.
#
# This code is inspired by the LlamaFactory.
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/cli.py
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


import sys
from functools import partial
from .webui.interface import run_web_ui
from .vocab.vocab_train import do_vocab_train
from .pretrain.default_trainer import do_pretrain
from .misc.misc import test_args
from .inference.inf import do_infer
from .sft.default_sft_trainer import do_sft

USAGE = (
    "-" * 67
    + "\n"
    + "| Usage:                                                          |\n"
    + "|   rainfallmodel pretrain: pretrain LLM                          |\n"
    + "|   rainfallmodel infer: interfence LLM                           |\n"
    + "|   rainfallmodel webui: launch WebUI of RainFallModel            |\n"
    + "|   rainfallmodel version: show version info of RainFallModel     |\n"
    + "-" * 67
)


def main():
    from .common.version import VERSION
    WELCOME = (
        "-" * 63
        + "\n"
        + f"| Welcome to RainFall Model, version {VERSION}"
        + " " * (25 - len(VERSION))
        + "|\n|"
        + " " * 61
        + "|\n"
        + "| Project page: https://github.com/rainfallclub/rainfallmodel |\n"
        + "-" * 63
    )

    COMMAND_MAP = {
        "sft": do_sft,
        "infer": do_infer,
        "misc": test_args,
        "pretrain": do_pretrain,
        "vocab": do_vocab_train,
        "webui": run_web_ui,
        "version": partial(print, WELCOME),
        "help": partial(print, USAGE),
    }

    
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    if command in COMMAND_MAP:
        COMMAND_MAP[command]()
    else:
        print(f"Unknown command, Please check: {command}.\n{USAGE}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
