from pathlib import Path
import random
import re
from typing import Dict, List
# ------------ Добавляем в sys.path:
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -------------

from validation.csv_dataset_export import save_spans_csv
from validation.csv_dataset_import import load_csv

# Построение соседей по раскладке ЙЦУКЕН
RUS_ROWS = [
    "ёйцукенгшщзхъ",
    "фывапролджэ",
    "ячсмитьбю",
]

# Смартфон, БЕЗ автокоррекции (сырая печать)
MOBILE_NO_AUTOCORR = {
    "typo_probs": {"sub": 0.85, "ins": 0.08, "del": 0.07},
    "prob": 0.02,   # ≈2% ошибок на символ — реалистично для телефона без коррекции
}

def _build_neighbors(rows: List[str]) -> Dict[str, List[str]]:
    neighbors: Dict[str, List[str]] = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            neigh = set()
            # Соседи в той же строке
            if c - 1 >= 0: neigh.add(row[c - 1])
            if c + 1 < len(row): neigh.add(row[c + 1])
            # Соседи в строке выше/ниже (с учётом "шахматного" сдвига)
            for rr in (r - 1, r + 1):
                if 0 <= rr < len(rows):
                    updown = rows[rr]
                    for dc in (-1, 0, 1):
                        cc = c + dc
                        if 0 <= cc < len(updown):
                            neigh.add(updown[cc])
            neigh.discard(ch)
            neighbors[ch] = sorted(neigh)
    # Добавим верхний регистр, чтобы сохранять регистр при подстановках
    full = dict(neighbors)
    for ch, neigh in neighbors.items():
        full[ch.upper()] = [n.upper() for n in neigh]
    return full

NEIGHBORS = _build_neighbors(RUS_ROWS)


RUS_LETTER_RE = re.compile(r"[А-Яа-яЁё]")

def _is_rus_letter(ch: str) -> bool:
    return bool(RUS_LETTER_RE.fullmatch(ch))

def _rand_neighbor(ch: str) -> str:
    arr = NEIGHBORS.get(ch)
    if arr:
        return random.choice(arr)
    # Если по каким-то причинам нет соседей — вернём исходную букву
    return ch

# --- Основная функция ---
def make_typos(
    text: str,
    prob: float = 0.12,
    typo_probs: Dict[str, float] = None,
    seed: int | None = None,
) -> str:
    """
    Вносит опечатки в русские слова с вероятностью `prob` на символ.
    Поддерживаемые типы опечаток:
      - 'sub'  (замена на соседнюю клавишу)
      - 'del'  (удаление символа)
      - 'ins'  (вставка соседней клавиши)
      - 'swap' (перестановка с соседним символом)
    `typo_probs` — распределение вероятностей типов ошибок (нормируется автоматически).
    """
    if seed is not None:
        random.seed(seed)

    if typo_probs is None:
        typo_probs = {"sub": 0.6, "del": 0.15, "ins": 0.15, "swap": 0.10}
    # Нормируем распределение
    total = sum(typo_probs.values())
    if total <= 0:
        typo_probs = {"sub": 1.0}
        total = 1.0
    weights = {k: v / total for k, v in typo_probs.items()}

    chars = list(text)
    i = 0
    out = []

    while i < len(chars):
        ch = chars[i]
        if _is_rus_letter(ch) and random.random() < prob:
            # Выбор типа опечатки
            r = random.random()
            acc = 0.0
            choice = "sub"
            for k, w in weights.items():
                acc += w
                if r <= acc:
                    choice = k
                    break

            if choice == "sub":
                out.append(_rand_neighbor(ch))

            elif choice == "del":
                # пропускаем символ (ничего не добавляем в out)
                pass

            elif choice == "ins":
                # вставим соседний символ и сам символ
                ins = _rand_neighbor(ch)
                # случайно до или после
                if random.random() < 0.5:
                    out.append(ins)
                    out.append(ch)
                else:
                    out.append(ch)
                    out.append(ins)

            elif choice == "swap":
                # Поменять местами с следующим русским символом, если он есть
                if i + 1 < len(chars) and _is_rus_letter(chars[i + 1]):
                    out.append(chars[i + 1])
                    out.append(ch)
                    i += 1  # доп. сдвиг, т.к. "съели" следующий символ
                else:
                    # если свап невозможен — fallback к замене
                    out.append(_rand_neighbor(ch))
        else:
            out.append(ch)
        i += 1

    return "".join(out)


if __name__ == "__main__":
    s = "Привет! Это тестовая строка с русскими буквами: ёж, ЙЦУКЕН, ящерица."
    result_str = make_typos(s, prob=0.05, seed=44, typo_probs=MOBILE_NO_AUTOCORR["typo_probs"])
    print(result_str)
    
    DATASETS_FOLDER = Path(__file__).resolve().parents[1] / "data"
    SRC_DATASETS = [DATASETS_FOLDER / "train_brands.csv", DATASETS_FOLDER / "train.csv"]
    RESULT_DATASET = DATASETS_FOLDER / "train_augmentated_all.csv"
    RETRIES = 5

    result_texts = []
    result_spans = []
    for i in range(RETRIES):
        for dataset in SRC_DATASETS:
            texts, spans = load_csv(dataset)
            for num, text in enumerate(texts):
                text = make_typos(text, prob=0.05, seed=i, typo_probs=MOBILE_NO_AUTOCORR["typo_probs"])
                result_texts.append(text)
                result_spans.append(spans[num])
    save_spans_csv(
        spans_per_row=result_spans,
        path=RESULT_DATASET,
        texts=result_texts,
        with_header=True)
