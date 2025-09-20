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

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer

from ..dataset.dataset_manager import get_sft_dataset
from transformers import Trainer, TrainingArguments
from ..model.model_manager import get_real_model_path

def get_full_ft_args(sft_conf:dict) -> TrainingArguments:
    """
    设置微调相关的参数
    """
    args = TrainingArguments(
        output_dir=sft_conf['output_dir'],
        per_device_train_batch_size=int(sft_conf['batch_size']),
        gradient_accumulation_steps=int(sft_conf['gradient_accumulation_steps']),
        logging_steps=int(sft_conf["logging_steps"]),
        bf16=torch.cuda.is_bf16_supported(), # 先设置为这样，后续再改
        fp16=not torch.cuda.is_bf16_supported(),
        num_train_epochs=int(sft_conf['epochs']),
        save_steps=int(sft_conf['save_steps']),
        learning_rate=float(sft_conf["learning_rate"]),
        lr_scheduler_type=sft_conf["lr_scheduler_type"],
        gradient_checkpointing=True,
        report_to="none",
    )
    return args


def do_full_ft(sft_conf:dict):
    """
    执行全量微调
    """

    # 第一步，加载分词器和模型
    model_path = sft_conf['model_path']
    real_model_path = get_real_model_path(model_path)
    model = AutoModelForCausalLM.from_pretrained(real_model_path)
    tokenizer = AutoTokenizer.from_pretrained(real_model_path)

    # 第二步，处理数据集
    train_dataset = get_sft_dataset(tokenizer, sft_conf)

    # 第三步，定义Trainer所需的参数
    args = get_full_ft_args(sft_conf)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)


    # 第四步，开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
