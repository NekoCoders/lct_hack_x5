from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification

model_id = "dreyk111/x5-ner-add-brands"
output_model_id = "nekocoders/x5-ner-final"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.push_to_hub(repo_id=output_model_id)

# Torch model
model_hf = AutoModelForTokenClassification.from_pretrained(model_id)
model_hf.push_to_hub(repo_id=output_model_id)

# ONNX model
model_ort = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
model_ort.save_pretrained("./x5-ner-onnx", push_to_hub=True, repository_id=output_model_id)


