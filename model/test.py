from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model_id = "bert-finetuned-ner/checkpoint-9198"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "молоко домик в деревне"

ner_results = nlp(example)
print(ner_results)
