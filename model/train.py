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

def make_transform(apply_typos: bool):
    """
    Возвращает функцию, которую Datasets вызовет на КАЖДОЙ выборке,
    уже во время обучения/валидации.
    """
def make_transform(apply_typos: bool):
    """
    Возвращает ф-ю, которую Datasets вызовет на КАЖДОМ батче
    прямо на лету при формировании даталоадера.
    """
    def _transform(batch):
        # batch["spans"] — List[List[str]], batch["labels"] — List[List[int]]
        words_batch  = batch["spans"]
        labels_batch = batch["labels"]

        input_ids_batch      = []
        attention_mask_batch = []
        label_ids_batch      = []

        for words, labels in zip(words_batch, labels_batch):
            # На train добавляем случайные опечатки
            if apply_typos:
                words = [default_make_typos(w) for w in words]

            # Токенизация списка слов одного примера
            tok = tokenizer(
                [words],                       # <-- батч из одного примера
                is_split_into_words=True,
                truncation=True,
            )
            # Берём нулевой элемент из батча
            input_ids      = tok["input_ids"][0]
            attention_mask = tok["attention_mask"][0]
            word_ids       = tok.word_ids(batch_index=0)

            aligned = align_labels_with_tokens(labels, word_ids)

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            label_ids_batch.append(aligned)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": label_ids_batch,
        }
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
    remove_unused_columns=False,  # чтобы не выкидывало 'spans' до трансформа
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
