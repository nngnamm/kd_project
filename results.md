# Experimental Results

## 1. Overview

The main goal of this experiment was to see whether knowledge distillation could produce a smaller BERT-based student model while keeping its classification performance reasonably close to the teacher model.

The experiments were done on the **GLUE SST-2** sentiment classification dataset. The same 6-layer student architecture was used for both experiments. The main difference was the training objective:

- **Knowledge Distillation (KD):** `alpha = 0.5`, `temperature = 4.0`
- **CE-only baseline:** `alpha = 0.0`

Both experiments used the same random seed (`42`), learning rate, batch size, number of epochs, warmup ratio, and other training settings. This was done so that the main variable being compared was the use of knowledge distillation.

---

## 2. Teacher and Student

The teacher model was:

`textattack/bert-base-uncased-SST-2`

The teacher contains **109,483,778 parameters** and achieved **92.43% validation accuracy** on SST-2.

For the student model, the original BERT encoder was reduced from 12 Transformer layers to 6 layers. The student contains **66,956,546 parameters**.

This gives a parameter reduction of approximately **38.8%** compared with the teacher.

The student still keeps the original BERT embedding layers, which is why removing half of the Transformer layers does not reduce the total number of parameters by exactly 50%.

---

## 3. Knowledge Distillation Result

The knowledge distillation model was trained with:

```text
alpha = 0.5
temperature = 4.0
seed = 42
epochs = 3
learning rate = 5e-5
warmup ratio = 0.1
```

The validation accuracy after each epoch was:

| Epoch | Validation Accuracy |
|------:|--------------------:|
| 1 | 91.06% |
| 2 | 89.91% |
| 3 | 90.14% |

The final saved KD model was the model from **epoch 3**, with a validation accuracy of **90.14%**.

Compared with the teacher's 92.43% accuracy, the student had an accuracy drop of **2.29 percentage points**.

The exact code version for this experiment is stored in Git under commit:

```text
3657b3e
```

and tagged as:

```text
kd-sst2-90.14
```

---

## 4. CE-only Baseline

To understand whether the improvement came from knowledge distillation itself, a second experiment was run using the same student model and training setup, but without the distillation component.

For the baseline:

```text
alpha = 0.0
temperature = 4.0
seed = 42
epochs = 3
learning rate = 5e-5
warmup ratio = 0.1
```

Since `alpha = 0.0`, the knowledge distillation term has no contribution to the final loss. The model is therefore trained using the normal cross-entropy loss with the hard SST-2 labels.

The validation accuracy after each epoch was:

| Epoch | Validation Accuracy |
|------:|--------------------:|
| 1 | 90.14% |
| 2 | 90.14% |
| 3 | 90.60% |

The final baseline accuracy was **90.60%**.

The exact code version for this experiment is stored in Git under commit:

```text
910eed1
```

and tagged as:

```text
ce-baseline-90.60
```

---

## 5. KD vs. CE Baseline

The two final results are:

| Model | Validation Accuracy |
|---|---:|
| Teacher | 92.43% |
| Student + KD | 90.14% |
| Student + CE only | 90.60% |

The CE-only baseline was **0.46 percentage points higher** than the KD model:

```text
90.60% - 90.14% = 0.46 percentage points
```

So, in this particular experiment, knowledge distillation did **not** produce a higher validation accuracy than training the same student model with only cross-entropy.

However, this difference is relatively small. The SST-2 validation set contains 872 examples, so a change of less than one percentage point can reasonably come from normal validation-set variation. Because of this, I would not consider this single experiment enough evidence to conclude that CE training is genuinely better than KD training.

A more reliable comparison would require repeating the experiments with multiple random seeds and comparing the average performance and variation.

---

## 6. Inference Benchmark

The teacher and student were also compared during inference.

The benchmark was performed on a single GPU using FP32 precision and batch size 1. The average latency was measured over 200 validation examples after 20 warm-up runs.

| Model | Parameters | Accuracy | Mean Latency |
|---|---:|---:|---:|
| Teacher | 109,483,778 | 92.43% | 8.78 ms |
| Student + KD | 66,956,546 | 90.14% | 4.59 ms |

The student reduced the parameter count by approximately **38.8%** and achieved a latency of **4.59 ms**, compared with **8.78 ms** for the teacher.

This corresponds to approximately a **1.92× speedup**:

```text
8.78 / 4.59 ≈ 1.92
```

The latency improvement is larger than the percentage reduction in parameters. This is expected because the student has only 6 Transformer layers instead of 12, which reduces the sequential depth of the model and the amount of computation performed during inference.

---

## 7. Discussion

The results show that the smaller student model can keep a relatively high level of sentiment classification performance while being noticeably cheaper to run.

The KD student achieved **90.14%** validation accuracy, which is only **2.29 percentage points** below the teacher. At the same time, it uses about **38.8% fewer parameters** and has approximately **1.92× lower inference latency** in the benchmark.

One interesting result is that the CE-only baseline reached **90.60%**, slightly higher than the KD model. This means that knowledge distillation did not provide an accuracy advantage in this specific configuration.

Looking at the training process, the KD model reached its highest validation accuracy after the first epoch at **91.06%**, then dropped to **89.91%** in the second epoch and finished at **90.14%**. This suggests that the model was already learning useful representations early in training and then showed some instability or overfitting as training continued.

The training loss also continued to decrease while the validation performance did not consistently improve. This is another sign that lower training loss does not necessarily mean better generalization.

Because only one seed was tested for each setting, these results should be treated as an initial experiment rather than a final statistical comparison. More runs with different seeds would make the conclusion much stronger.

---

## 8. Main Findings

The main findings from the current experiments are:

1. A 6-layer BERT student can achieve around **90% validation accuracy** on SST-2 while being substantially smaller than the 12-layer teacher.

2. The KD student achieved **90.14%**, compared with **92.43%** for the teacher.

3. The student reduced the parameter count by approximately **38.8%**.

4. The measured inference latency improved from **8.78 ms to 4.59 ms**, giving an approximately **1.92× speedup**.

5. The CE-only baseline achieved **90.60%**, which was **0.46 percentage points higher** than the KD model in this single-seed experiment.

6. The current experiment therefore does not show a clear accuracy benefit from knowledge distillation, although the comparison should be repeated with multiple seeds before drawing a stronger conclusion.

---

## 9. Reproducibility

The two main experiments are tracked using Git tags:

```text
kd-sst2-90.14
ce-baseline-90.60
```

The corresponding commits are:

```text
KD:
3657b3e

CE baseline:
910eed1
```

The trained models and experiment logs were also backed up separately.

This makes it possible to go back to the exact code version used for each experiment instead of relying only on the reported numbers.

