
import logging
import time

log = logging.getLogger("model_runner")

log.info("Start of loading transformers library")
from transformers import AutoTokenizer
from transformers import pipeline
log.info("End of loading transformers library")
from optimum.onnxruntime import ORTModelForTokenClassification
# ------------ Добавляем в sys.path:
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -------------
from server.interface import Entity
from model.postprocessing import splitted_bio_spans_from_ents

MODEL_ID = "nekocoders/x5-ner-final"

log = logging.getLogger(__file__)


class NERPipeline:
    def __init__(self):
        self.load_model()

    def load_model(self):  # FIXME: must be in module "model"?
        log.info("Start model loading")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = ORTModelForTokenClassification.from_pretrained(MODEL_ID, provider="CUDAExecutionProvider")

        self.pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        log.info("Model loaded")

    def infer_model(self, texts: list[str]) -> list[list[Entity]]:  # FIXME: must be in module "model"?
        start_time = time.perf_counter()  # TODO: move to decorator

        result_ents_for_texts = self.pipe(texts)
        response_entities_for_texts = []
        for i, result_ents in enumerate(result_ents_for_texts):
            result_spans_for_text = splitted_bio_spans_from_ents(text=texts[i], ents=result_ents, model_id=MODEL_ID)
            response_entities = [Entity(start_index=start, end_index=end, entity=label)
                                for start, end, label in result_spans_for_text]
            response_entities_for_texts.append(response_entities)

        duration = time.perf_counter() - start_time
        log.info("infer_model executed in '%.3f s', number of texts: '%d'", duration, len(texts))
        return response_entities_for_texts


if __name__ == "__main__":
    pipe = NERPipeline()
    result = pipe.infer_model(["Молоко простоквашино"])
    print(result)
