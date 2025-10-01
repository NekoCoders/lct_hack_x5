import csv
from pathlib import Path
import re
# ------------ Добавляем в sys.path:
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -------------
from validation.csv_dataset_export import save_spans_csv

DATA_FOLDER = Path(__file__).resolve().parents[1] / "data"
SOURCE = DATA_FOLDER / "openfoodfacts_export.csv"
DST_RAW_DATASET = DATA_FOLDER / "brands.csv"
DST_RESULT_DATASET = DATA_FOLDER / "train_brands.csv"
brands = set()
with open(SOURCE, "r", encoding="utf-8") as f:
    iter_items = csv.DictReader(f, delimiter="\t")
    for item in iter_items:
        brands.add(item["brands"])

with open(DST_RAW_DATASET, "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows([(re.sub(r'&[a-z]+;', '', brand),) for brand in brands if brand])


with open(DST_RAW_DATASET, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    brands = []
    for row in reader:
        brands.extend(row)
    brands = filter(lambda i: i is not None and i != "None", brands)
    brands = list(set(brands))


spans_per_row = []
for brand_text in brands:
    words = filter(lambda i: i, brand_text.split(" "))
    spans_of_text = []
    last_word_end = 0
    for i, word in enumerate(words):
        prefix = "B-" if i == 0 else "I-"
        spans_of_text.append((last_word_end+1, last_word_end+len(word), f"{prefix}BRAND"))
        last_word_end = last_word_end+len(word)
    start, end, label = spans_of_text[-1]
    spans_of_text[-1] = (start, end+1, label)
    spans_per_row.append(spans_of_text)

save_spans_csv(
    spans_per_row=spans_per_row,
    path=DST_RESULT_DATASET,
    texts=brands,
    with_header=True)
