import ast
from pathlib import Path
from torch.utils.data.dataset import Dataset

from datasets import load_dataset, Features, Sequence, Value, ClassLabel


def _convert_sample(sample):
    text = sample["sample"]
    annotation = ast.literal_eval(sample["annotation"])

    spans = []
    labels = []
    for start, end, label in annotation:
        span = text[start:end]
        spans.append(span)
        labels.append(label)

    return {"spans": spans, "labels": labels}


def load_custom_dataset():
    dataset_path = str(Path(__file__).parent.parent / "data/train.csv")
    dataset = load_dataset("csv", data_files={"train": dataset_path}, delimiter=";")

    labels = [
        "O",
        "B-TYPE",
        "I-TYPE",
        "B-BRAND",
        "I-BRAND",
        "B-PERCENT",
        "I-PERCENT",
        "B-VOLUME",
        "I-VOLUME",
    ]

    dataset = dataset.map(_convert_sample, remove_columns=["sample", "annotation"])

    dataset = dataset.cast(
        Features(
            {
                "spans": Sequence(Value("string")),
                "labels": Sequence(ClassLabel(names=labels)),
            }
        )
    )

    return dataset
