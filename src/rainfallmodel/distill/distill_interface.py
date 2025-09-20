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

from typing import TYPE_CHECKING, Any, Optional
from ..common.parser import read_args
from .distill_white_box_full import do_distill_wb_full
from .distill_white_box_lora import do_distill_wb_lora
from .distill_config import get_user_distill_conf


def do_distill(args: Optional[dict[str, Any]] = None) -> None:
    """
    微调主流程，支持全量微调和LoRA微调
    """

    print("distill begin!!")
    # 第一步，读取配置并进行转换
    user_conf = read_args(args)
    distill_conf = get_user_distill_conf(user_conf)

    distill_type = distill_conf['distill_type']

    if 'full' == distill_type:
        do_distill_wb_full(distill_conf)
    elif 'lora' == distill_type:
        do_distill_wb_lora(distill_conf)
    else:
        print("unknown distill_type, please check your config!!")

    print("distill done!!")

