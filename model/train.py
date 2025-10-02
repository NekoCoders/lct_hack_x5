from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)

from dataset import (
    ID2LABEL,
    LABEL2ID,
    align_labels_with_tokens,
    load_raw_datasets,
    tokenize_datasets,
)
from augmentation import default_make_typos
from metrics import compute_metrics

base_model = "xlm-roberta-large"
output_model = "x5-ner-with-augmentation-in-flight"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForTokenClassification.from_pretrained(
    base_model,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

raw_datasets = load_raw_datasets()
tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer)

def make_transform(apply_typos: bool):
    """
    Возвращает функцию, которую Datasets вызовет на КАЖДОЙ выборке,
    уже во время обучения/валидации.
    """
    def _transform(example):
        words = example["spans"]           # список "слов" (токенов по словам)
        labels = example["labels"]         # BIO-метки на уровне слов

        # На train добавляем случайные опечатки внутрь слов.
        if apply_typos:
            # Применяем по словам, чтобы не склеивать их
            words = [default_make_typos(w) for w in words]

        # Токенизируем "список слов"
        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
        )

        # Выравниваем метки по субтокенам
        word_ids = tokenized.word_ids()
        tokenized["labels"] = align_labels_with_tokens(labels, word_ids)

        return tokenized
    return _transform

train_dataset = raw_datasets["train"].with_transform(make_transform(apply_typos=True))
eval_dataset = raw_datasets["test"].with_transform(make_transform(apply_typos=False))

args = TrainingArguments(
    output_model,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    resume_from_checkpoint=output_model,
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
trainer.train(resume_from_checkpoint=False)

trainer.push_to_hub()
