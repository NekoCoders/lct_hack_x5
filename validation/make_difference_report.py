from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
# ------------ Добавляем в sys.path:
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(PROJECT_ROOT)
# -------------
from validation.csv_dataset_import import load_csv

SpanType = Tuple[int, int, str]  # (start, end, label)

def _type_without_prefix(label: str) -> str:
    if not label or label == 'O':
        return 'O'
    return label.split('-', 1)[-1] if '-' in label else label

def _covers(span: Tuple[int,int,str], s: int, e: int) -> bool:
    a, b, _ = span
    return a <= s and e <= b and s < e

def _label_for_segment(spans: List[SpanType], s: int, e: int) -> str:
    # ищем спан, который полностью покрывает сегмент; если несколько, берём с максимальным перекрытием (обычно один)
    best = 'O'
    best_ov = 0
    for a, b, lab in spans:
        ov = max(0, min(b, e) - max(a, s))
        if ov > best_ov and ov > 0:
            best_ov = ov
            best = lab
    return best if best_ov > 0 else 'O'

def _segment_tokens(text: str, segs: List[Tuple[int,int]]) -> List[str]:
    return [text[s:e] for s, e in segs]

def _join_with_bars(tokens: List[str]) -> str:
    return "|".join(tokens)

def _expand_by_spaces(text: str, s: int, e: int, left_words: int = 1, right_words: int = 1) -> Tuple[int,int]:
    # расширяем (s,e) к ближайшим пробельным границам, добавив по N "слов" слева/справа
    L = s
    R = e
    # влево
    for _ in range(left_words):
        while L > 0 and text[L-1].isspace() is False:
            L -= 1
        while L > 0 and text[L-1].isspace():
            L -= 1
    # вправо
    for _ in range(right_words):
        while R < len(text) and text[R].isspace() is False:
            R += 1
        while R < len(text) and text[R].isspace():
            R += 1
    return max(0,L), min(len(text),R)

def print_bio_diff_report(
    texts: List[str],
    gold_spans_all: List[List[SpanType]],
    pred_spans_all: List[List[SpanType]],
    max_examples_per_type: int = 20
) -> None:
    """
    Печатает отчёт по двум наборам BIO-спанов (символьные оффсеты):
      - <N> ошибок токенизации (границы различаются при совпадающем TYPE),
      - <M> ошибок определения сущности (BIO/TYPE различаются).
    """
    assert len(texts) == len(gold_spans_all) == len(pred_spans_all)

    tokenization_examples = []
    labeling_examples = []

    total_tokenization_errors = 0
    total_labeling_errors = 0

    for text, gold_spans, pred_spans in zip(texts, gold_spans_all, pred_spans_all):
        # 1) построим объединённые границы
        bounds = {0, len(text)}
        for s, e, _ in gold_spans:
            bounds.add(int(s)); bounds.add(int(e))
        for s, e, _ in pred_spans:
            bounds.add(int(s)); bounds.add(int(e))
        cuts = sorted(b for b in bounds if 0 <= b <= len(text))
        segments = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1) if cuts[i] < cuts[i+1]]

        # 2) последовательность «микро-сегментов» с метками gold/pred
        micro = []
        for s, e in segments:
            g = _label_for_segment(gold_spans, s, e)
            p = _label_for_segment(pred_spans, s, e)
            micro.append((s, e, g, p))

        # ---- Ошибки определения сущности (там, где метки не равны) ----
        for s, e, g, p in micro:
            if g != p:
                total_labeling_errors += 1
                # сформируем короткую цитату вокруг сегмента
                L, R = _expand_by_spaces(text, s, e, left_words=1, right_words=1)
                quote = text[L:R].replace("\n", " ").strip()
                under = " " * (s+1) + "^" * max(1, e - s)
                labeling_examples.append({
                    "quote": text,
                    "explain": f"{p} вместо {g}",
                    "caret": under
                })

        # ---- Ошибки токенизации ----
        # идея: найдём пересекающиеся «кластеры» одинакового TYPE в gold и pred
        # для каждого типа TYPE соберём пересечения, посчитаем число уникальных спанов gold/pred в пересечении
        # если числа различаются — ошибка токенизации; сохраним пример с токенами через |
        # сгруппируем по типам
        by_type_gold: Dict[str, List[Tuple[int,int]]] = defaultdict(list)
        by_type_pred: Dict[str, List[Tuple[int,int]]] = defaultdict(list)
        for s, e, g, p in micro:
            gt, pt = _type_without_prefix(g), _type_without_prefix(p)
            if gt != 'O':
                by_type_gold[gt].append((s, e))
            if pt != 'O':
                by_type_pred[pt].append((s, e))

        # для каждого типа создадим объединённые интервалы пересечения
        for typ in set(by_type_gold.keys()).intersection(by_type_pred.keys()):
            # объединённая маска сегментов, где и gold, и pred = этот typ
            mask = [(s, e) for (s, e, g, p) in micro if _type_without_prefix(g) == typ and _type_without_prefix(p) == typ]
            if not mask:
                continue
            # слепим соседние маски в крупные блоки
            blocks = []
            cur_s, cur_e = None, None
            for s, e in mask:
                if cur_s is None:
                    cur_s, cur_e = s, e
                elif s == cur_e:
                    cur_e = e
                else:
                    blocks.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            if cur_s is not None:
                blocks.append((cur_s, cur_e))

            # по каждому блоку посчитаем число исходных токенов gold/pred,
            # то есть число спанов из gold/pred, которые попали в блок
            for bs, be in blocks:
                gold_tokens = sorted({(max(bs, s), min(be, e)) for (s, e, lab) in gold_spans if _type_without_prefix(lab) == typ and max(bs, s) < min(be, e)})
                pred_tokens = sorted({(max(bs, s), min(be, e)) for (s, e, lab) in pred_spans if _type_without_prefix(lab) == typ and max(bs, s) < min(be, e)})

                if not gold_tokens or not pred_tokens:
                    continue  # на всякий случай

                if len(gold_tokens) != len(pred_tokens) or gold_tokens != pred_tokens:
                    total_tokenization_errors += 1
                    gold_str = _join_with_bars(_segment_tokens(text, gold_tokens))
                    pred_str = _join_with_bars(_segment_tokens(text, pred_tokens))
                    tokenization_examples.append({
                        "type": typ,
                        "gold": gold_str,
                        "pred": pred_str
                    })

    # ---- Печать отчёта ----
    print(f"{total_tokenization_errors} ошибок токенизации (считая предложения):")
    for ex in tokenization_examples[:max_examples_per_type]:
        print(f"\"{ex['gold']}\" вместо \"{ex['pred']}\"")  # при желании меняй порядок
    print()

    print(f"{total_labeling_errors} ошибок определения сущности (считая предложения):")
    for ex in labeling_examples[:max_examples_per_type]:
        print(f"\"{ex['quote']}\"  - {ex['explain']}")
        print(ex["caret"])


if __name__ == "__main__":
    TRUE_CSV = Path("data") / "submission_orig.csv"
    PRED_CSV = Path("data") / "submission.csv"

    true_texts, true_spans = load_csv(TRUE_CSV)
    pred_texts, pred_spans = load_csv(PRED_CSV)
    true_texts = true_texts[:len(pred_texts)]
    true_spans = true_spans[:len(pred_texts)]
    print_bio_diff_report(true_texts, true_spans, pred_spans, max_examples_per_type=20)
