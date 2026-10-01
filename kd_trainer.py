import torch
import torch.nn.functional as F
from transformers import Trainer


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=4.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model.to(self.args.device)
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        student_outputs = model(**model_inputs)
        student_logits = student_outputs.logits.float()

        with torch.no_grad():
            teacher_logits = self.teacher(**model_inputs).logits.float()

        T = self.temperature

        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T**2)

        hard_loss = F.cross_entropy(student_logits, labels)

        loss = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss

        return (loss, student_outputs) if return_outputs else loss
