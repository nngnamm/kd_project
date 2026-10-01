# Knowledge Distillation for BERT on SST-2

This project explores **knowledge distillation (KD)** for NLP classification using BERT and the **GLUE SST-2 sentiment analysis dataset**.

The main idea is simple: instead of using the full BERT model for inference, we train a smaller BERT-based student model and try to transfer useful knowledge from a larger teacher model.

The project also includes a **CE-only baseline** so that we can check whether knowledge distillation actually helps compared with normal supervised training.

---

## Project Goal

The main goal is to answer two questions:

1. Can a smaller BERT model keep reasonably good sentiment classification performance?
2. Does knowledge distillation help the smaller model compared with training it directly with cross-entropy?

For this experiment, the teacher uses the standard 12-layer BERT architecture, while the student keeps only the bottom 6 Transformer layers.

---

## Dataset

The project uses the **GLUE SST-2 (Stanford Sentiment Treebank)** dataset.

SST-2 is a binary sentiment classification task:

- `0` → negative
- `1` → positive

The dataset used in the experiments contains:

| Split | Samples |
|---|---:|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

The validation set was used for evaluation because the labels of the official test set are not available for normal local evaluation.

The dataset is loaded directly with Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("glue", "sst2")
```

---

## Models

### Teacher

The teacher model is:

```text
textattack/bert-base-uncased-SST-2
```

It is a fine-tuned BERT-base model with 12 Transformer encoder layers.

Teacher parameters:

```text
109,483,778
```

Validation accuracy:

```text
92.43%
```

The teacher is frozen during student training, so its parameters are not updated.

### Student

The student starts from:

```text
bert-base-uncased
```

and uses only the bottom 6 Transformer encoder layers.

The student has:

```text
66,956,546 parameters
```

This is approximately:

```text
38.8% fewer parameters
```

than the teacher.

The parameter reduction is not exactly 50% because the model still keeps components such as the BERT embedding layers and classification head.

---

## Knowledge Distillation

The student is trained using a combination of two losses:

- **Cross-entropy loss** from the true SST-2 labels
- **Knowledge distillation loss** from the teacher's predictions

The loss used in the experiment is:

```text
L = αT² KL(teacher || student)
    + (1 - α) CE(student, labels)
```

where:

- `α` controls the contribution of distillation
- `T` is the temperature used to soften the probability distributions
- `KL` is the Kullback-Leibler divergence
- `CE` is the normal cross-entropy loss

For the main KD experiment:

```text
alpha = 0.5
temperature = 4.0
seed = 42
```

A temperature greater than 1 makes the output distribution softer, which can expose more information about the teacher's relative confidence between classes.

---

## Baseline

To make the comparison fair, a second experiment was run with:

```text
alpha = 0.0
```

This removes the distillation contribution from the loss and leaves normal cross-entropy training.

The important part is that the student architecture and the main training settings were kept the same.

### Main training settings

```text
Seed:              42
Epochs:            3
Learning rate:     5e-5
Warmup ratio:      0.1
Weight decay:      0.01
Per-device batch:  16
Effective batch:   32 (2 GPUs)
Precision:         FP16
```

This makes the baseline a useful ablation because the main difference is whether knowledge distillation is used.

---

## Results

### Validation Accuracy

| Model | Accuracy |
|---|---:|
| Teacher | **92.43%** |
| Student + KD | **90.14%** |
| Student + CE only | **90.60%** |

The KD model finished at **90.14%**, while the CE-only baseline reached **90.60%**.

So in this single-seed experiment, the baseline was:

```text
0.46 percentage points higher
```

than the KD model.

This does **not** give strong evidence that CE is genuinely better than KD. The validation set only contains 872 examples, so a difference of less than one percentage point can be caused by normal variation. More runs with different random seeds would be needed for a stronger comparison.

---

## Training Progress

### KD experiment

| Epoch | Validation Accuracy |
|---:|---:|
| 1 | 91.06% |
| 2 | 89.91% |
| 3 | 90.14% |

The best validation accuracy actually happened after the first epoch. The performance then dropped and slightly recovered by the final epoch.

This suggests some instability or overfitting during later training. The training loss continued to decrease even though validation accuracy did not keep improving, which is a useful reminder that lower training loss does not always mean better generalization.

### CE baseline

| Epoch | Validation Accuracy |
|---:|---:|
| 1 | 90.14% |
| 2 | 90.14% |
| 3 | 90.60% |

The baseline improved slightly during the final epoch.

---

## Inference Benchmark

The teacher and student were also benchmarked during inference.

The benchmark used:

- Single GPU
- FP32
- Batch size 1
- 20 warm-up runs
- 200 validation sentences for the timing measurement

| Model | Parameters | Accuracy | Mean Latency |
|---|---:|---:|---:|
| Teacher | 109,483,778 | 92.43% | 8.78 ms |
| Student + KD | 66,956,546 | 90.14% | 4.59 ms |

The student therefore achieved approximately:

```text
1.92× speedup
```

compared with the teacher.

This is interesting because the parameter count decreased by about 38.8%, while the measured latency improved by about 47.7%.

The two numbers are not expected to match exactly. Removing Transformer layers reduces the sequential depth of the model as well as the amount of computation performed during inference. Actual latency is also affected by memory access, GPU execution, and framework overhead.

---

## Project Structure

```text
kd_project/
│
├── data.py
├── models.py
├── kd_trainer.py
├── train.py
├── benchmark.py
├── check_teacher.py
├── inspect_data.py
└── decode_demo.py
```

### Main files

**`data.py`**  
Loads and prepares the SST-2 dataset and tokenizer inputs.

**`models.py`**  
Contains the teacher/student model setup.

**`kd_trainer.py`**  
Contains the custom Hugging Face Trainer used to calculate the knowledge distillation loss.

**`train.py`**  
Runs the student training experiment and contains the main experiment settings such as seed, alpha, temperature, and output directory.

**`benchmark.py`**  
Measures model size, accuracy, and inference latency.

**`check_teacher.py`**  
Checks the teacher model and its validation performance.

**`inspect_data.py`**  
Used to inspect the dataset and confirm that the inputs and labels are loaded correctly.

**`decode_demo.py`**  
Small utility for inspecting and decoding model/tokenizer inputs.

---

## Reproducibility

The two main experiments are tracked using Git tags.

### Knowledge Distillation

```text
Commit: 3657b3e
Tag:    kd-sst2-90.14
Alpha:  0.5
T:      4.0
Seed:   42
Result: 90.14% validation accuracy
```

### CE Baseline

```text
Commit: 910eed1
Tag:    ce-baseline-90.60
Alpha:  0.0
T:      4.0
Seed:   42
Result: 90.60% validation accuracy
```

This means the exact code version used for each experiment can be recovered later even after more changes are made to the repository.

---

## Running the Project

The original experiments were trained on **Kaggle Notebooks using 2× NVIDIA Tesla T4 GPUs**.

The general workflow is:

```text
Load SST-2
      ↓
Tokenize sentences
      ↓
Load frozen teacher
      ↓
Build 6-layer student
      ↓
Train student
      ↓
Evaluate on validation set
      ↓
Benchmark accuracy / parameters / latency
```

For the KD experiment, the main settings are:

```text
alpha = 0.5
temperature = 4.0
seed = 42
```

For the CE baseline:

```text
alpha = 0.0
temperature = 4.0
seed = 42
```

---

## Current Conclusion

The current experiment shows that the 6-layer student can retain a relatively high level of SST-2 performance while being noticeably smaller and faster than the teacher.

The student reduced the parameter count by approximately **38.8%** and achieved around **1.92× lower inference latency** in the benchmark.

However, the current results do not show a clear accuracy advantage from knowledge distillation. The KD model achieved **90.14%**, while the CE-only baseline achieved **90.60%**.

Since this comparison was performed with only one random seed, the next useful experiment is to repeat both settings with additional seeds and compare the average accuracy and variation rather than relying on a single run.

---

## Experiment Backup

The trained models and experiment logs were also backed up separately after training.

The backup contains:

```text
student_final/
student_ce_baseline/
log_kd_sst2.json
log_ce_baseline.json
kd_project/
```

This allows the trained models and experiment records to be preserved separately from the Git source-code history.

