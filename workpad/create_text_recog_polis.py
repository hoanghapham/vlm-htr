
import torch
from datasets import load_dataset

# data = load_dataset("HuggingFaceM4/DocumentVQA")

polis = load_dataset("Riksarkivet/goteborgs_poliskammare_fore_1900", trust_remote_code=True, name="text_recognition")

# hovratt = load_dataset("Riksarkivet/gota_hovratt_seg", trust_remote_code=True)
