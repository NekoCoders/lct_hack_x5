from pathlib import Path

import pandas as pd

# ------------ Добавляем в sys.path:
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(PROJECT_ROOT)
# -------------
from model.interface import SpanType
from model.tokenizer import word_spans_by_tokenizer

def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    # строгая проверка пересечения отрезков по символам
    return max(a_start, b_start) < min(a_end, b_end)

def splitted_bio_spans_from_ents(text: str, ents: list[dict]) -> list[SpanType]:
    # Словные токены (по токенайзеру) с их символьными оффсетами
    word_spans = word_spans_by_tokenizer(text)
    bio_spans: list[SpanType] = []
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
        # Вариант без добавления префиксов:
        # e_start, e_end = int(ent["start"]), int(ent["end"])
        # label = str(ent["entity_group"]).strip()
        # bio_spans.append((e_start, e_end, label))
    return fill_empty_spans_between_ents(word_spans=word_spans, ents=bio_spans)

def fill_empty_spans_between_ents(word_spans: list[tuple[int, int]], ents: list[SpanType]) -> list[SpanType]:
    """SpanType: (start_char, end_char, entity_type) """
    ent_spans = {(ent[0], ent[1]) for ent in ents}
    empty_spans = set(word_spans) - ent_spans
    empty_ents = [(span[0], span[1], "O") for span in empty_spans]
    return sorted([*ents, *empty_ents], key=lambda ent: ent[0])
