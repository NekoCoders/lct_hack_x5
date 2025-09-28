
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
# ------------ Добавляем в sys.path:
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(PROJECT_ROOT)
# -------------
from model.postprocessing import splitted_bio_spans_from_ents

MODEL_ID = "lotusbro/x5-ner"

log = logging.getLogger("model_runner")


def load_model():  # FIXME: must be in module "model"?
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

    pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    log.info("Model loaded")
    return pipe


def infer_model(text):  # FIXME: must be in module "model"?
    global pipe
    if "pipe" not in globals():
        pipe = load_model()

    result_ents = pipe(text)
    result_spans = splitted_bio_spans_from_ents(text=text, ents=result_ents, model_id=MODEL_ID)
    return result_spans


if __name__ == "__main__":
    result = infer_model("Молоко простоквашино")
    print(result)
