import os

os.environ["WANDB_DISABLED"] = "true"

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

from data import get_data
from models import get_teacher, count_params

FINAL_DIR = "/kaggle/working/student_final"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_accuracy(model, dataset, collator, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    correct = 0
    total = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(**batch).logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def measure_latency(model, tokenizer, sentences, warmup=20):
    encoded = [
        tokenizer(s, return_tensors="pt", truncation=True, max_length=128).to(device)
        for s in sentences
    ]
    for enc in encoded[:warmup]:
        model(**enc)
    torch.cuda.synchronize()

    times = []
    for enc in encoded:
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(**enc)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times)), float(np.std(times))


def main():
    _, val_ds, tokenizer, collator = get_data()
    sentences = load_dataset("glue", "sst2")["validation"]["sentence"][:200]

    teacher = get_teacher().to(device).eval()
    student = AutoModelForSequenceClassification.from_pretrained(FINAL_DIR)
    student = student.to(device).eval()

    results = {}
    for name, model in [("Teacher (12L)", teacher), ("Student (6L)", student)]:
        lat_mean, lat_std = measure_latency(model, tokenizer, sentences)
        results[name] = {
            "params": count_params(model),
            "acc": evaluate_accuracy(model, val_ds, collator),
            "lat": lat_mean,
            "lat_std": lat_std,
        }

    t = results["Teacher (12L)"]
    s = results["Student (6L)"]

    line = "+" + "-" * 28 + "+" + "-" * 20 + "+" + "-" * 20 + "+"
    print(line)
    print(f"| {'Metric':<26} | {'Teacher (12L)':<18} | {'Student (6L)':<18} |")
    print(line)
    print(f"| {'Total Parameters':<26} | {t['params']:<18,} | {s['params']:<18,} |")
    print(
        f"| {'SST-2 Val Accuracy (%)':<26} | {t['acc'] * 100:<18.2f} | {s['acc'] * 100:<18.2f} |"
    )
    print(f"| {'Latency (ms, batch=1)':<26} | {t['lat']:<18.2f} | {s['lat']:<18.2f} |")
    print(line)

    print()
    print(f"Parameter reduction: {(1 - s['params'] / t['params']) * 100:.1f}%")
    print(f"Accuracy retained:   {s['acc'] / t['acc'] * 100:.2f}% of teacher")
    print(f"Accuracy drop:       {(t['acc'] - s['acc']) * 100:.2f} points")
    print(f"Latency speedup:     {t['lat'] / s['lat']:.2f}x")


if __name__ == "__main__":
    main()
