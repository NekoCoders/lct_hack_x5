
import logging
log = logging.getLogger("model_runner")

log.info("Start of loading transformers library")
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
log.info("End of loading transformers library")
# ------------ Добавляем в sys.path:
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -------------
from model.interface import SpanType
from model.postprocessing import splitted_bio_spans_from_ents

MODEL_ID = "lotusbro/x5-ner"


def load_model():  # FIXME: must be in module "model"?
    log.info("Start model loading")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

    pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    log.info("Model loaded")
    return pipe


def infer_model(texts: list[str]) -> list[SpanType]:  # FIXME: must be in module "model"?
    global pipe
    if "pipe" not in globals():
        pipe = load_model()

    result_ents_for_texts = pipe(texts)
    result_spans_for_texts = []
    for i, result_ents in enumerate(result_ents_for_texts):
        result_spans_for_text = splitted_bio_spans_from_ents(text=texts[i], ents=result_ents, model_id=MODEL_ID)
        result_spans_for_texts.append(result_spans_for_text)
    return result_spans_for_texts


if __name__ == "__main__":
    result = infer_model("Молоко простоквашино")
    print(result)
