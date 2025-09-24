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
    LABEL_NAMES,
    load_custom_dataset,
    tokenize_datasets,
)
from metrics import compute_metrics

base_model = "xlm-roberta-large"
output_model = "x5-ner"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForTokenClassification.from_pretrained(
    base_model,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

raw_datasets = load_custom_dataset()

tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer)

tokenized_datasets = tokenized_datasets["train"].train_test_split(
    test_size=0.1, seed=42
)

args = TrainingArguments(
    output_model,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    resume_from_checkpoint=output_model,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
trainer.train(resume_from_checkpoint=True)

trainer.push_to_hub()
