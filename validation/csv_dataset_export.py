import csv
from typing import Optional
from validation.interface import SpanType


def save_spans_csv(
    spans_per_row: list[list[SpanType]],
    path: str,
    texts: Optional[list[str]] = None,
    sep: str = ";",
    with_header: bool = True,
) -> None:
    """
    Сохраняет список списков спанов в CSV.
    - spans_per_row: для каждой строки датасета — список (start, end, label)
    - texts: опционально — соответствующие тексты (пойдут в 1-ю колонку)
    - sep: разделитель колонок (по умолчанию ';')
    - with_header: писать ли заголовок
    Формат сохраняемой второй колонки: repr(list_of_spans),
    т.е. ровно "[(0, 4, 'B-TYPE'), (5, 8, 'I-TYPE'), ...]",
    что удобно читать через ast.literal_eval потом.
    """
    if texts is not None and len(texts) != len(spans_per_row):
        raise ValueError("Длины texts и spans_per_row должны совпадать.")

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=sep, quoting=csv.QUOTE_MINIMAL)
        if with_header:
            writer.writerow(["sample", "annotation"] if texts is not None else ["annotation"])

        for i, spans in enumerate(spans_per_row):
            spans_str = repr(spans)  # гарантирует формат [(0, 4, 'B-...'), ...]
            if texts is not None:
                writer.writerow([texts[i], spans_str])
            else:
                writer.writerow([spans_str])
