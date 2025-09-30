from pathlib import Path
from collections import defaultdict
import pandas as pd
# ------------ Добавляем в sys.path:
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -------------

from validation.csv_dataset_import import load_csv
from model.interface import BaseStatsPerEntity, SpanType


def calculate_tp_fp_fn(pred_spans: list[list[SpanType]], true_spans: list[list[SpanType]], filter_out_empty_spans: True) -> tuple[int, int, int]:
    tp: BaseStatsPerEntity = defaultdict(int) 
    fp: BaseStatsPerEntity = defaultdict(int)
    fn: BaseStatsPerEntity = defaultdict(int)

    # цикл по текстам:
    for y_true_sequence, y_pred_sequence in zip(true_spans, pred_spans):
        if filter_out_empty_spans:
            y_true_sequence = filter(lambda span: span[2] != "O", y_true_sequence)
            y_pred_sequence = filter(lambda span: span[2] != "O", y_pred_sequence)
        gold_entities = set(y_true_sequence)      # {(start, end, label), ...}
        predicted_entities = set(y_pred_sequence) # {(start, end, label), ...}

        exact_entity_matches = gold_entities & predicted_entities  # точные совпадения типа и границ

        for _, _, label in exact_entity_matches:
            tp[label] += 1
        for _, _, label in gold_entities - exact_entity_matches:
            fn[label] += 1
        for _, _, label in predicted_entities - exact_entity_matches:
            fp[label] += 1
    return tp, fp, fn


def calculate_full_stats_from_base(tp: BaseStatsPerEntity, fp: BaseStatsPerEntity, fn: BaseStatsPerEntity):
    entity_labels_sorted = sorted(set(tp) | set(fp) | set(fn))

    metrics_rows = []
    macro_precision_sum = macro_recall_sum = macro_f1_sum = 0.0
    macro_label_count = 0
    macro_total_support = 0

    for label in entity_labels_sorted:
        precision_value = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall_value = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1_value = (2 * precision_value * recall_value / (precision_value + recall_value)) if (precision_value + recall_value) else 0.0
        support_value = tp[label] + fn[label]  # число gold-сущностей данного типа

        metrics_rows.append({
            "entity": label,
            "precision": round(precision_value, 4),
            "recall": round(recall_value, 4),
            "f1": round(f1_value, 4),
            "support": int(support_value),
        })

        if support_value > 0:
            macro_precision_sum += precision_value
            macro_recall_sum += recall_value
            macro_f1_sum += f1_value
            macro_label_count += 1
            macro_total_support += support_value

    # macro avg по типам
    if macro_label_count > 0:
        macro_f1 = round(macro_f1_sum / macro_label_count, 4)
        metrics_rows.append({
            "entity": "macro avg",
            "precision": round(macro_precision_sum / macro_label_count, 4),
            "recall": round(macro_recall_sum / macro_label_count, 4),
            "f1": macro_f1,
            "support": int(macro_total_support),
        })
    return macro_f1, metrics_rows


def calculate_macro_f1_from_spans(pred_spans: list[SpanType], true_spans: list[SpanType]) -> float:
    base_stats = calculate_tp_fp_fn(pred_spans=pred_spans, true_spans=true_spans)
    macro_f1, _ = calculate_full_stats_from_base(*base_stats)
    return macro_f1


if __name__ == "__main__":
    TRUE_CSV = Path("data") / "test_small_gold.csv"
    PRED_CSV = Path("data") / "test_small_predict.csv"

    true_texts, true_spans = load_csv(TRUE_CSV)
    pred_texts, pred_spans = load_csv(PRED_CSV)
    base_stats = calculate_tp_fp_fn(pred_spans=pred_spans, true_spans=true_spans)
    macro_f1, full_stats = calculate_full_stats_from_base(*base_stats)

    df_out = pd.DataFrame(full_stats, columns=["entity", "precision", "recall", "f1", "support"])
    print("Precision - насколько правильные сущности мы нашли?")
    print("Recall - какой % сущностей мы нашли?")
    print("Support - количество сущностей в gold")
    print(df_out.to_string(index=False))
