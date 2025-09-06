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


from transformers import Trainer, TrainingArguments
from swanlab.integration.transformers import SwanLabCallback
from ..vocab.info import get_tokenizer
from ..dataset.dataset_manager import get_pretrain_dataset
from ..model.model_manager import get_pretrain_model
from .pretrain_config import get_user_pretrain_conf

import torch
from typing import TYPE_CHECKING, Any, Optional
from ..common.parser import read_args
from transformers import DataCollatorForLanguageModeling

def get_pretrain_args(pretrain_conf:dict) -> TrainingArguments:
    """
    获取预训练相关的参数
    """
    args = TrainingArguments(
        output_dir= pretrain_conf["output_dir"],
        num_train_epochs=int(pretrain_conf['epochs']),
        do_train=True, # 直接设置为True
        do_eval=True, # 这里是否允许不做eval呢？
        per_device_train_batch_size=int(pretrain_conf['batch_size']),
        gradient_accumulation_steps=int(pretrain_conf['gradient_accumulation_steps']),
        logging_steps=int(pretrain_conf["logging_steps"]),
        report_to='none',
        save_total_limit=int(pretrain_conf['save_total_limit']),
        save_steps=int(pretrain_conf['save_steps']),
        bf16=torch.cuda.is_bf16_supported(), # 先设置为这样，后续再改
        fp16=not torch.cuda.is_bf16_supported(),
        learning_rate=float(pretrain_conf["learning_rate"]),
        lr_scheduler_type=pretrain_conf["lr_scheduler_type"],
        dataloader_num_workers=int(pretrain_conf["dataloader_num_workers"]),
        dataloader_pin_memory=True,
        # 验证相关部分
        eval_strategy=pretrain_conf["eval_strategy"],
        eval_steps=int(pretrain_conf["eval_steps"]),
        save_safetensors=pretrain_conf['save_safetensors'])
    return args


def get_callback(pretrain_conf:dict) -> list:
    """
    回调部分，暂时先实现swanlab的监控上报
    """

    if not pretrain_conf['use_swanlab']:
        return []

    swanlab_callback = SwanLabCallback(
        project= pretrain_conf['swanlab_project_name'],
        experiment_name= pretrain_conf['swanlab_experiment_name'],
        config={
            "platform": "rainfall_model_factory",
        }
    )

    return [swanlab_callback]


def do_pretrain(args: Optional[dict[str, Any]] = None) -> None:
    """
    预训练主流程，目前先简单实现
    待完成部分:
    1).日志部分
    2).上报部分，tensorboard、wandb等
    3).分布式部分
    4).模型定制诉求
    5).多种数据集支持
    6).其他加速技术
    7).历史记录部分，这块需要统一定义
    """

    # 第一步，读取配置并进行转换
    user_conf = read_args(args)
    pretrain_conf = get_user_pretrain_conf(user_conf)

    # 第二步，加载模型和分词
    tokenizer_path = user_conf["tokenizer_path"]
    tokenizer = get_tokenizer(tokenizer_path)
    model = get_pretrain_model(user_conf)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # 第三步，处理数据集
    train_dataset, eval_dataset = get_pretrain_dataset(tokenizer, user_conf)

    # 第四步，定义Trainer所需的参数
    args = get_pretrain_args(pretrain_conf)   
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    callbacks = get_callback(pretrain_conf)

    # 第五步，开始训练
    trainer = Trainer(
        callbacks=callbacks,
        model=model, 
        args=args, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer, 
        data_collator=data_collator)
    trainer.train()

    print("Train success!!")


