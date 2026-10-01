from data import get_data

train_ds, val_ds, tokenizer, collator = get_data()

ids = train_ds[0]["input_ids"]
print(tokenizer.convert_ids_to_tokens(ids))

batch = collator([train_ds[0], train_ds[2]])
print(batch["input_ids"])
print(batch["attention_mask"])
print(batch["labels"])
