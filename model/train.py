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
    load_raw_datasets,
)
from augmentation import make_transform_augments
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

train_dataset = raw_datasets["train"].with_transform(make_transform_augments(tokenizer, apply_typos=True))
eval_dataset = raw_datasets["test"].with_transform(make_transform_augments(tokenizer, apply_typos=False))

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
