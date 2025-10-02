import ast
from pathlib import Path
from torch.utils.data.dataset import Dataset

from datasets import load_dataset, Features, Sequence, Value, ClassLabel, concatenate_datasets, DatasetDict


LABEL_NAMES = [
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

ID2LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_NAMES)}


def _convert_sample(sample):
    text = sample["sample"]
    annotation = ast.literal_eval(sample["annotation"])  # List[Tuple[int, int, str]]

    spans = []
    labels = []
    offsets = []
    for start, end, label in annotation:
        span = text[start:end]
        spans.append(span)
        labels.append(label)
        offsets.append((int(start), int(end)))

    return {"spans": spans, "labels": labels, "text": text, "offsets": offsets}


def load_custom_dataset(data_filename: str):
    file_path = str(Path(__file__).parent.parent / "data" / data_filename)
    dataset = load_dataset("csv", data_files={"train": file_path}, delimiter=";")
    features = Features(
        {
            "spans": Sequence(Value("string")),
            "labels": Sequence(ClassLabel(names=LABEL_NAMES)),
            "text": Value("string"),
            "offsets": Sequence(Sequence(Value("int32"))),  # [[start, end], ...]
        }
    )

    dataset = dataset.map(
        _convert_sample,
        remove_columns=["sample", "annotation"],
        features=features,
    )

    return dataset


def load_raw_datasets():
    dataset_main_train = load_custom_dataset("train.csv")
    dataset_brands = load_custom_dataset("train_brands.csv")
    # делим только основной train.csv
    split_main = dataset_main_train["train"].train_test_split(test_size=0.1, seed=42)
    # объединяем brands с train-частью
    full_train = concatenate_datasets([split_main["train"], dataset_brands["train"]])

    raw_datasets = DatasetDict({
        "train": full_train,
        "test": split_main["test"]
    })
    return raw_datasets


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_datasets(raw_datasets: Dataset, tokenizer):
    def tokenize_and_align_labels(sample):
        tokenized_inputs = tokenizer(
            sample["spans"], truncation=True, is_split_into_words=True
        )
        all_labels = sample["labels"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    return tokenized_datasets
