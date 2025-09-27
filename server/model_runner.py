
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def load_model():
    model_id = "lotusbro/x5-ner"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)

    pipe = pipeline("ner", model=model, tokenizer=tokenizer)
    return pipe

def infer_model(text):
    global pipe
    if "pipe" not in globals():
        pipe = load_model()

    result = pipe(text)
    return result
