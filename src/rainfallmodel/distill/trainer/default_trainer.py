from transformers import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

class Default_Trainer(Trainer):
    def __init__(self, 
                student_model,
                teacher_model,
                args,
                train_dataset,
                eval_dataset,
                tokenizer,
                data_collator,
                reduction = "sum",
                alpha=0.5, 
                temperature=3.0):
        super().__init__(
            student_model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer
        )
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.reduction=reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_loss(self, model, inputs, **kwargs):
        # 获取教师和学生输出
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        student_outputs = model(**inputs)
        
        # 交叉熵损失
        ce_loss = student_outputs.loss
        
        # KL散度损失
        teacher_logits = teacher_outputs.logits / self.temperature
        log_probs = torch.log_softmax(student_outputs.logits / self.temperature, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (teacher_probs * (teacher_log_probs - log_probs)).sum(-1)
        kl_loss = kl.masked_fill_(inputs['labels'].eq(-100), 0.0).sum()

        print(f"ce_loss:{ce_loss.detach()}, kl_loss:{kl_loss.detach()}")
        # print(f"alpha: {self.alpha}, temp:{self.temperature}, reduction:{self.reduction}")
        
        # 组合损失
        loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        
        return  loss

