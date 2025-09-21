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
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from transformers import TrainingArguments
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from ..model.model_manager import get_real_model_path
from ..dataset.dataset_manager import get_sft_dataset
from .trainer.default_trainer import Default_Trainer
from swanlab.integration.transformers import SwanLabCallback

def get_distill_args(distill_conf:dict):
    """
    设置蒸馏相关的参数
    """
    args = TrainingArguments(
        output_dir=distill_conf['output_dir'],
        per_device_train_batch_size=int(distill_conf['batch_size']),
        gradient_accumulation_steps=int(distill_conf['gradient_accumulation_steps']),
        logging_steps=int(distill_conf["logging_steps"]),
        bf16=torch.cuda.is_bf16_supported(), # 先设置为这样，后续再改
        fp16=not torch.cuda.is_bf16_supported(),
        num_train_epochs=int(distill_conf['epochs']),
        save_steps=int(distill_conf['save_steps']),
        learning_rate=float(distill_conf["learning_rate"]),
        lr_scheduler_type=distill_conf["lr_scheduler_type"],
        gradient_checkpointing=True,
        report_to="none",
    )
    return args 

def get_callback(distill_conf:dict) -> list:
    """
    回调部分，暂时先实现swanlab的监控上报
    """

    if not distill_conf['use_swanlab']:
        return []

    swanlab_callback = SwanLabCallback(
        project= distill_conf['swanlab_project_name'],
        experiment_name= distill_conf['swanlab_experiment_name'],
        config={
            "platform": "rainfall_model_factory",
        }
    )

    return [swanlab_callback]

def do_distill_wb_full(distill_conf:dict):
    
    # 第一步，加载学生模型、分词器和教师模型
    student_model_path = distill_conf['student_model_path']
    real_student_model_path = get_real_model_path(student_model_path)
    student_model = AutoModelForCausalLM.from_pretrained(real_student_model_path).to(distill_conf['device'])
    tokenizer = AutoTokenizer.from_pretrained(real_student_model_path)
    teacher_model_path = distill_conf['teacher_model_path']
    real_teacher_model_path = get_real_model_path(teacher_model_path)
    teacher_model = AutoModelForCausalLM.from_pretrained(real_teacher_model_path).to(distill_conf['device'])
    teacher_model.eval()
    
    # 第二步，获取数据集
    train_dataset = get_sft_dataset(tokenizer, distill_conf)

    # 第三步，获取Trainer所需的参数
    args = get_distill_args(distill_conf)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    callbacks = get_callback(distill_conf)

    # 第四步，开始训练
    trainer = Default_Trainer(student_model=student_model,
                        teacher_model=teacher_model,
                        args=args, 
                        train_dataset=train_dataset, 
                        eval_dataset=None,
                        tokenizer=tokenizer, 
                        data_collator=data_collator,
                        reduction=distill_conf["reduction"], 
                        alpha=float(distill_conf["loss_alpha"]),
                        temperature=float(distill_conf["temperature"]),
                        callbacks=callbacks
                        )
    trainer.train()
   
    
      
    

























