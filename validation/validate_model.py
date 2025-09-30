
from datetime import datetime
from pathlib import Path
from functools import lru_cache

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd

# ------------ Добавляем в sys.path:
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -------------

from model.dataset import ID2LABEL, load_custom_dataset
from validation.make_difference_report import REPORTS_DIR, print_bio_diff_report
from validation.csv_dataset_export import save_spans_csv
from validation.csv_dataset_import import load_csv
from model.interface import SpanType
from validation.validate_spans import calculate_full_stats_from_base, calculate_tp_fp_fn
from model.postprocessing import splitted_bio_spans_from_ents

@lru_cache(maxsize=1)
def _get_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)

@lru_cache(maxsize=1)
def _get_pipeline(model_id: str):
    tokenizer = _get_tokenizer(model_id=model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def predict_spans_by_model_bio(texts: list[str], model_id: str) -> list[list[SpanType]]:
    """
    Возвращает BIO-спаны на уровне словных токенов токенайзера:
    [(start_char, end_char, 'B-<TYPE>/I-<TYPE>'), ...]
    """
    nlp = _get_pipeline(model_id)
    all_results: list[list[SpanType]] = []

    for text in tqdm(texts):
        # Агрегированные сущности от пайплайна (уже склеенные в фразы)
        ents = nlp(text)  # каждый ent: {start, end, entity_group, ...}
        result_spans = splitted_bio_spans_from_ents(text=text, ents=ents, model_id=model_id)
        all_results.append(result_spans)

    return all_results


def load_test_part_of_train_dataset(model_id: str) -> tuple[list, list]:
    def _convert_dataset_to_spanlists(ds) -> list[list[SpanType]]:
        all_spans: list[list[SpanType]] = []
        all_texts: list[str] = []
        for row in ds["test"]:
            spans: list[SpanType] = []
            all_texts.append(row["text"])
            for (start, end), label_id in zip(row["offsets"], row["labels"]):
                spans.append((int(start), int(end), ID2LABEL[int(label_id)]))
            all_spans.append(spans)
        return all_texts, all_spans

    raw_datasets = load_custom_dataset()
    tokenized_datasets = raw_datasets["train"].train_test_split(
        test_size=0.1, seed=42  # Параметры должны совпадать с model/train.py! TODO: вынести в одно место
    )  # {"train": [], "test": []} Structure: see in load_custom_dataset() func

    return _convert_dataset_to_spanlists(tokenized_datasets)


if __name__ == "__main__":
    DOCUMENTS_TO_PROCESS = None
    DIFF_EXAMPLES_NUMBER = 50
    USE_TRAIN_TEXTS = True
    MODEL_IDS_ORDERED = ["lotusbro/x5-ner", "lotusbro/x5-ner-ru"]
    MODEL = MODEL_IDS_ORDERED[0]

    if USE_TRAIN_TEXTS:
        true_texts, true_spans = load_test_part_of_train_dataset(model_id=MODEL)
        true_dataset = "test-part-of-train"
    else:
        TRUE_CSV = Path("data") / "submission_orig.csv"
        true_dataset = TRUE_CSV.name
        true_texts, true_spans = load_csv(TRUE_CSV)

    true_texts = true_texts[:DOCUMENTS_TO_PROCESS]
    true_spans = true_spans[:DOCUMENTS_TO_PROCESS]

    PREDICTED_CSV = Path("data") / "submission.csv"
    pred_spans = predict_spans_by_model_bio(texts=true_texts, model_id=MODEL)

    save_spans_csv(
    spans_per_row=pred_spans,
    path=PREDICTED_CSV,
    texts=true_texts,
    sep=";",
    )

    base_stats = calculate_tp_fp_fn(pred_spans=pred_spans, true_spans=true_spans, filter_out_empty_spans=False)
    macro_f1, full_stats = calculate_full_stats_from_base(*base_stats)

    REPORTS_DIR.mkdir(exist_ok=True)
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_%H-%M")
    with open(REPORTS_DIR / f"{datetime_str}_{true_dataset}-{PREDICTED_CSV.name}.txt", "w", encoding="utf-8") as f:
        import sys
        sys.stdout = f

        df_out = pd.DataFrame(full_stats, columns=["entity", "precision", "recall", "f1", "support"])
        print("Precision - насколько правильные сущности мы нашли?")
        print("Recall - какой % сущностей мы нашли?")
        print("Support - количество сущностей в gold")
        print(df_out.to_string(index=False))

        print_bio_diff_report(true_texts, true_spans, pred_spans, max_examples_per_type=50)
