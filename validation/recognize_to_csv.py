
from pathlib import Path

# ------------ Добавляем в sys.path:
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(PROJECT_ROOT)
# -------------

from validation.csv_dataset_export import save_spans_csv
from validation.csv_dataset_import import load_csv
from validation.validate_model import predict_spans_by_model_bio


if __name__ == "__main__":
    MODEL = "nekocoders/x5-ner-final"
    TRUE_CSV = Path("data") / "submission_orig.csv"
    true_dataset = TRUE_CSV.name
    true_texts, true_spans = load_csv(TRUE_CSV)

    PREDICTED_CSV = Path("data") / f"submission-{MODEL.replace('/', '-')}.csv"
    pred_spans = predict_spans_by_model_bio(texts=true_texts, model_id=MODEL)

    save_spans_csv(
    spans_per_row=pred_spans,
    path=PREDICTED_CSV,
    texts=true_texts,
    sep=";",
    )
