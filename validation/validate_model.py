
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
from validation.csv_dataset_import import load_csv
from validation.interface import SpanType
from validation.validate_spans import calculate_full_stats_from_base, calculate_tp_fp_fn
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

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
    TRUE_CSV = Path("data") / "submission.csv"
    true_texts, true_spans = load_csv(TRUE_CSV)
    pred_spans = predict_spans_by_model(texts=true_texts)

    base_stats = calculate_tp_fp_fn(pred_spans=pred_spans, true_spans=true_spans)
    macro_f1, full_stats = calculate_full_stats_from_base(*base_stats)

    df_out = pd.DataFrame(full_stats, columns=["entity", "precision", "recall", "f1", "support"])
    print("Precision - насколько правильные сущности мы нашли?")
    print("Recall - какой % сущностей мы нашли?")
    print("Support - количество сущностей в gold")
    print(df_out.to_string(index=False))
