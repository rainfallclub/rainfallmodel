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

from ..common.resource import get_dataset_config, get_root_path

def check_need_remote(dataset_path:str) -> tuple[bool, str]:
    """
    检测是否需要链接远程仓库，如果不需要则返回绝对路径
    """
    dataset_dict = get_dataset_config()
    exist_in_dataset_config = dataset_path in dataset_dict
    if not exist_in_dataset_config:
        return False, dataset_path
    
    dataset_conf = dataset_dict[dataset_path]
    if dataset_conf['type'] == 'local_file_relative_path':
        final_path = get_root_path() + '/' + dataset_conf['file_path']
        return False, final_path
    
    return True, dataset_conf['repo']



