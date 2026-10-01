from datasets import load_dataset

ds = load_dataset("glue", "sst2")

print(ds)

for i in range(5):
    row = ds["train"][i]
    print(row["label"], "|", row["sentence"])

val = ds["validation"]
print(val[0])
print(sum(val["label"]), "positive out of", len(val))

print(ds["test"][0])
