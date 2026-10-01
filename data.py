import os

os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

TEACHER_NAME = "textattack/bert-base-uncased-SST-2"


def get_data():
    raw_ds = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=128)

    tokenized = raw_ds.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["sentence", "idx"])
    tokenized = tokenized.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenized["train"], tokenized["validation"], tokenizer, collator


if __name__ == "__main__":
    train_ds, val_ds, tokenizer, collator = get_data()
    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))
    print("First example:", train_ds[0])
