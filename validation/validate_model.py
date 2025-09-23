
from pathlib import Path
from functools import lru_cache
import random

import pandas as pd

# ------------ Добавляем в sys.path:
import sys
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(PROJECT_ROOT)
# -------------
from validation.csv_dataset_export import save_spans_csv
from validation.csv_dataset_import import load_csv
from model.interface import SpanType
from validation.validate_spans import calculate_full_stats_from_base, calculate_tp_fp_fn
from model.tokenizer import word_spans_by_tokenizer
from model.postprocessing import fill_empty_spans_between_ents, splitted_bio_spans_from_ents


MODEL_ID = "lotusbro/x5-ner"

@lru_cache(maxsize=1)
def _get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)

@lru_cache(maxsize=1)
def _get_pipeline():
    tok = _get_tokenizer()
    mdl = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    return pipeline("ner", model=mdl, tokenizer=tok, aggregation_strategy="simple")

def predict_spans_by_model_bio(texts: list[str]) -> list[list[SpanType]]:
    """
    Возвращает BIO-спаны на уровне словных токенов токенайзера:
    [(start_char, end_char, 'B-<TYPE>/I-<TYPE>'), ...]
    """
    nlp = _get_pipeline()
    all_results: list[list[SpanType]] = []

    for text in tqdm(texts):
        # Агрегированные сущности от пайплайна (уже склеенные в фразы)
        ents = nlp(text)  # каждый ent: {start, end, entity_group, ...}
        result_spans = splitted_bio_spans_from_ents(text=text, ents=ents)
        all_results.append(result_spans)

    return all_results


if __name__ == "__main__":
    DOCUMENTS_TO_PROCESS = 5000
    SHUFFLE_TEXTS = False

    TRUE_CSV = Path("data") / "submission_orig.csv"
    PREDICTED_CSV = Path("data") / "submission.csv"
    true_texts, true_spans = load_csv(TRUE_CSV)
    if SHUFFLE_TEXTS:
        pairs = list(zip(true_texts, true_spans))
        random.shuffle(pairs)
        true_texts, true_spans = zip(*pairs)

    true_texts = true_texts[:DOCUMENTS_TO_PROCESS]
    true_spans = true_spans[:DOCUMENTS_TO_PROCESS]
    pred_spans = predict_spans_by_model_bio(texts=true_texts)

    save_spans_csv(
    spans_per_row=pred_spans,
    path=PREDICTED_CSV,
    texts=true_texts,
    sep=";",
    )

    base_stats = calculate_tp_fp_fn(pred_spans=pred_spans, true_spans=true_spans, filter_out_empty_spans=False)
    macro_f1, full_stats = calculate_full_stats_from_base(*base_stats)

    df_out = pd.DataFrame(full_stats, columns=["entity", "precision", "recall", "f1", "support"])
    print("Precision - насколько правильные сущности мы нашли?")
    print("Recall - какой % сущностей мы нашли?")
    print("Support - количество сущностей в gold")
    print(df_out.to_string(index=False))
