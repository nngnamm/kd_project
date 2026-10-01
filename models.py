import torch.nn as nn
from transformers import AutoModelForSequenceClassification

TEACHER_NAME = "textattack/bert-base-uncased-SST-2"
STUDENT_BASE = "bert-base-uncased"
NUM_STUDENT_LAYERS = 6


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_teacher():
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_NAME)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def get_student():
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_BASE, num_labels=2
    )
    student.bert.encoder.layer = nn.ModuleList(
        student.bert.encoder.layer[:NUM_STUDENT_LAYERS]
    )
    student.config.num_hidden_layers = NUM_STUDENT_LAYERS
    return student


if __name__ == "__main__":
    teacher = get_teacher()
    student = get_student()
    print("Teacher layers:", len(teacher.bert.encoder.layer))
    print("Student layers:", len(student.bert.encoder.layer))
    print("Teacher params:", count_params(teacher))
    print("Student params:", count_params(student))
    print("Student config layers:", student.config.num_hidden_layers)
