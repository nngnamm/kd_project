import os

os.environ["WANDB_DISABLED"] = "true"

import numpy as np
from transformers import TrainingArguments, set_seed

from data import get_data
from models import get_teacher, get_student
from kd_trainer import DistillationTrainer

SEED = 42
SMOKE_TEST = True
OUTPUT_DIR = "/kaggle/working/kd_sst2"
FINAL_DIR = "/kaggle/working/student_final"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def main():
    set_seed(SEED)

    train_ds, val_ds, tokenizer, collator = get_data()
    if SMOKE_TEST:
        train_ds = train_ds.select(range(2000))

    teacher = get_teacher()
    student = get_student()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1 if SMOKE_TEST else 3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )

    trainer = DistillationTrainer(
        model=student,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        teacher_model=teacher,
        temperature=4.0,
        alpha=0.5,
    )

    print("Effective train batch size:", trainer.args.train_batch_size)

    trainer.train()

    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print("Saved student to", FINAL_DIR)


if __name__ == "__main__":
    main()
