
from pathlib import Path
from typing import List, Tuple
from functools import lru_cache

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
from validation.interface import SpanType
from validation.validate_spans import calculate_full_stats_from_base, calculate_tp_fp_fn

SpanType = Tuple[int, int, str]

MODEL_ID = "lotusbro/x5-ner"

@lru_cache(maxsize=1)
def _get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)

@lru_cache(maxsize=1)
def _get_pipeline():
    tok = _get_tokenizer()
    mdl = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    return pipeline("ner", model=mdl, tokenizer=tok, aggregation_strategy="simple")

def _word_spans_by_tokenizer(text: str) -> List[Tuple[int, int]]:
    """
    Возвращает список [(start_char, end_char)] для словных токенов,
    определённых препроцессором fast-токенайзера (с учётом пунктуации).
    """
    tok = _get_tokenizer()
    enc = tok(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = enc["offset_mapping"]
    # Для fast-токенайзеров доступна группировка в "слова"
    # через word_ids(). Несколько сабтокенов одного слова -> один словный спан.
    word_ids = enc.word_ids()  # может быть None для некоторых токенайзеров
    if word_ids is None:
        # fallback: берем offsets как есть (сабтокены)
        # и будем считать "слово" == сабтокен
        return [(s, e) for (s, e) in offsets if not (s == 0 and e == 0)]

    spans_by_word = []
    cur_id = None
    cur_start, cur_end = None, None

    for (s, e), wid in zip(offsets, word_ids):
        if wid is None or (s == 0 and e == 0):
            continue
        if cur_id is None:
            # старт первого слова
            cur_id = wid
            cur_start, cur_end = s, e
        elif wid == cur_id:
            # продолжаем то же слово (склеиваем сабтокены)
            cur_end = e
        else:
            # закончили предыдущее слово
            spans_by_word.append((cur_start, cur_end))
            # начинаем новое
            cur_id = wid
            cur_start, cur_end = s, e

    if cur_id is not None:
        spans_by_word.append((cur_start, cur_end))

    return spans_by_word

def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    # строгая проверка пересечения отрезков по символам
    return max(a_start, b_start) < min(a_end, b_end)

def predict_spans_by_model_bio(texts: List[str]) -> list[list[SpanType]]:
    """
    Возвращает BIO-спаны на уровне словных токенов токенайзера:
    [(start_char, end_char, 'B-<TYPE>/I-<TYPE>'), ...]
    """
    nlp = _get_pipeline()
    all_results: List[List[SpanType]] = []

    for text in tqdm(texts):
        # Агрегированные сущности от пайплайна (уже склеенные в фразы)
        ents = nlp(text)  # каждый ent: {start, end, entity_group, ...}

        # Словные токены (по токенайзеру) с их символьными оффсетами
        word_spans = _word_spans_by_tokenizer(text)

        bio_spans: List[SpanType] = []

        # Для каждой сущности – найдём покрытые словные токены
        for ent in ents:
            e_start, e_end = int(ent["start"]), int(ent["end"])
            label = str(ent["entity_group"]).strip()
            picked_indices = [
                i for i, (ws, we) in enumerate(word_spans)
                if _overlap(ws, we, e_start, e_end)
            ]
            if not picked_indices:
                continue  # на всякий случай, если препроцессор "порезал" иначе

            for j, idx in enumerate(picked_indices):
                ws, we = word_spans[idx]
                prefix = "B-" if j == 0 else "I-"
                bio_spans.append((ws, we, f"{prefix}{label}"))
        all_results.append(bio_spans)

    return all_results


def predict_spans_by_model(texts: list[str]) -> list[list[SpanType]]:
    model_id = "lotusbro/x5-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)

    # ВАЖНО: агрегация в сущности, а не по-токенно
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    recognized_results_per_text = []
    for text in texts:
        ner_results = nlp(text)
        # list[tuple[int, int, str]]: (start_char, end_char, label)
        spans: list[SpanType] = [
            (int(ent["start"]), int(ent["end"]), str(ent["entity_group"]))
            for ent in ner_results
        ]
        recognized_results_per_text.append(spans)
    return recognized_results_per_text


if __name__ == "__main__":
    TRUE_CSV = Path("data") / "submission_orig.csv"
    PREDICTED_CSV = Path("data") / "submission.csv"
    true_texts, true_spans = load_csv(TRUE_CSV)
    # DOCUMENTS_TO_PROCESS = 3000
    # true_texts = true_texts[1000:DOCUMENTS_TO_PROCESS]
    # true_spans = true_spans[1000:DOCUMENTS_TO_PROCESS]
    pred_spans = predict_spans_by_model_bio(texts=true_texts)

    save_spans_csv(
    spans_per_row=pred_spans,
    path=PREDICTED_CSV,
    texts=true_texts,
    sep=";",
    )

    base_stats = calculate_tp_fp_fn(pred_spans=pred_spans, true_spans=true_spans)
    macro_f1, full_stats = calculate_full_stats_from_base(*base_stats)

    df_out = pd.DataFrame(full_stats, columns=["entity", "precision", "recall", "f1", "support"])
    print("Precision - насколько правильные сущности мы нашли?")
    print("Recall - какой % сущностей мы нашли?")
    print("Support - количество сущностей в gold")
    print(df_out.to_string(index=False))
