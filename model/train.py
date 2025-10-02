from torch import nn
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    XLMRobertaForTokenClassification
)
from transformers.trainer_callback import EarlyStoppingCallback

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
model: XLMRobertaForTokenClassification = AutoModelForTokenClassification.from_pretrained(
    base_model,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# print(model)
model.dropout = torch.nn.Dropout(0.2, inplace=False)
freeze_modules = []
freeze_modules += [model.roberta.embeddings]
# freeze_modules += model.roberta.encoder.layer[:-12]
for module in freeze_modules:
    for param in module.parameters(): 
        param.requires_grad = False

raw_datasets = load_raw_datasets()

train_dataset = raw_datasets["train"].with_transform(make_transform_augments(tokenizer, apply_typos=True))
eval_dataset = raw_datasets["test"].with_transform(make_transform_augments(tokenizer, apply_typos=False))

training_args = TrainingArguments(
    output_model,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    max_grad_norm=1.0,
    gradient_accumulation_steps=2,
    dataloader_drop_last=False,
    gradient_checkpointing=False,
    remove_unused_columns=False,  # чтобы не выкидывало 'spans' до трансформа
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

trainer.push_to_hub()
