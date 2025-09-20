# Copyright 2025 the LlamaFactory team. and the RainFallModel team.
#
# This code is inspired by the LlamaFactory.
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/webui/manager.py
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


from collections.abc import Generator
from typing import TYPE_CHECKING
from .runner_vocab  import VocabRunner
from .runner_pretrain import PretrainRunner
from .runner_sft import SftRunner
from .runner_infer_base import InferBaseRunner
from .runner_infer_chat import InferChatRunner
from .runner_infer_compare import InferCompareRunner
from .runner_distill import DistillRunner


if TYPE_CHECKING:
    from gradio.components import Component

class Manager:
    r"""A class to manage all the gradio components in Web UI."""

    def __init__(self) -> None:
        self._id_to_elem: dict[str, Component] = {}
        self._elem_to_id: dict[Component, str] = {}
        self.vocab_runner = VocabRunner(self)
        self.pretrain_runner = PretrainRunner(self)
        self.sft_runner = SftRunner(self)
        self.infer_base_runner = InferBaseRunner(self)
        self.infer_chat_runner = InferChatRunner(self)
        self.infer_compare_runner = InferCompareRunner(self)
        self.distill_runner = DistillRunner(self)

    def get_lang(self) -> str:
        r"""Support Chinese Only Current"""
        return "zh"

    def add_elems(self, tab_name: str, elem_dict: dict[str, "Component"]) -> None:
        r"""Add elements to manager."""
        for elem_name, elem in elem_dict.items():
            elem_id = f"{tab_name}.{elem_name}"
            self._id_to_elem[elem_id] = elem
            self._elem_to_id[elem] = elem_id

    def get_elem_list(self) -> list["Component"]:
        r"""Return the list of all elements."""
        return list(self._id_to_elem.values())

    def get_elem_iter(self) -> Generator[tuple[str, "Component"], None, None]:
        r"""Return an iterator over all elements with their names."""
        for elem_id, elem in self._id_to_elem.items():
            yield elem_id.split(".")[-1], elem

    def get_elem_by_id(self, elem_id: str) -> "Component":
        r"""Get element by id."""
        return self._id_to_elem[elem_id]

    def get_id_by_elem(self, elem: "Component") -> str:
        r"""Get id by element."""
        return self._elem_to_id[elem]

    def get_base_elems(self) -> set["Component"]:
        r"""Get the base elements that are commonly used."""
        return {
            self._id_to_elem["top.lang"],
        }    