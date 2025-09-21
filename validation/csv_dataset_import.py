import ast
import pandas as pd

from validation.interface import SpanType

def load_csv(path: str) -> tuple[list[str], list[list[SpanType]]]:
    """
    Ожидается CSV, где:
      - 1-я колонка: текст
      - 2-я колонка: строка списком кортежей [(start, end, 'B-XXX'), ...]
    """
    df = pd.read_csv(path, header=0, sep=";")
    texts = df.iloc[:, 0].astype(str).tolist()
    spans_col = df.iloc[:, 1].fillna("").astype(str).tolist()
    spans = []
    for s in spans_col:
        s = s.strip()
        spans.append(ast.literal_eval(s) if s else [])
    return texts, spans
