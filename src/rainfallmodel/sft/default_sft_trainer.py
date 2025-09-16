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
from .full_ft import do_full_ft
from .lora_ft import do_lora_ft
from .sft_config import get_user_sft_conf

def do_sft(args: Optional[dict[str, Any]] = None) -> None:
    """
    微调主流程，目前仅支持全量微调
    """

    print("sft begin!!")
    # 第一步，读取配置并进行转换
    user_conf = read_args(args)
    sft_conf = get_user_sft_conf(user_conf)

    ft_type = sft_conf['ft_type']

    if 'full' == ft_type:
        do_full_ft(sft_conf)
    elif 'lora' == ft_type:
        do_lora_ft(sft_conf)
    else:
        print("unknown ft_type, please check your config!!")

    print("sft done!!")






