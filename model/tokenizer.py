from typing import List, Tuple
from functools import lru_cache


# ------------ Добавляем в sys.path:
import sys
import os
from transformers import AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -------------

SpanType = Tuple[int, int, str]

@lru_cache(maxsize=1)
def _get_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)

def word_spans_by_tokenizer(text: str, model_id: str) -> List[Tuple[int, int]]:
    """
    Возвращает список [(start_char, end_char)] для словных токенов,
    определённых препроцессором fast-токенайзера (с учётом пунктуации).
    """
    tok = _get_tokenizer(model_id=model_id)
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
