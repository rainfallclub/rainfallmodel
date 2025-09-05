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

from transformers import  PreTrainedModel
from .classical_model.default_llama_model import get_llama_model
from .model_config import get_user_model_config


def get_pretrain_model(user_conf:dict) -> PreTrainedModel:
    """
    暂时仅支持llama，后续会完善更加复杂的需求
    """

    model_conf = get_user_model_config(user_conf)
    return get_llama_model(model_conf)




