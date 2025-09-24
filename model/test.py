from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model_id = "lotusbro/x5-ner"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "сыр брест сливочный"

ner_results = nlp(example)
for span in ner_results:
    print(span)
