import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

NAME = "textattack/bert-base-uncased-SST-2"

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME)
model.eval()

val = load_dataset("glue", "sst2")["validation"]

correct = 0
for row in val:
    enc = tok(row["sentence"], return_tensors="pt", truncation=True)
    with torch.no_grad():
        pred = model(**enc).logits.argmax(-1).item()
    correct += int(pred == row["label"])

print("Teacher accuracy:", correct / len(val))
print("id2label:", model.config.id2label)
