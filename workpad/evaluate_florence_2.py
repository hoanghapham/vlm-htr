#%%
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from htrflow.evaluate import CER, WER

cer = CER()
wer = WER()
#%%
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = PROJECT_DIR / "models/florence-2-base-ft-vqa/"
remote_model_path = "microsoft/Florence-2-base-ft"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(remote_model_path, trust_remote_code=True)

#%%
data_stream = load_dataset("HuggingFaceM4/DocumentVQA", split="train", streaming=True)
data = list(data_stream.take(1100))


# data_files={
#         "train": "data/train-00000-of-00038.parquet",
#         "test": "data/test-00000-of-00017.parquet",
#         "validation": "data/validation-00000-of-00017.parquet"
#     }

#%%
sample_idx = 1070
print(data[sample_idx]["question"])
data[sample_idx]["image"]
#%%

prompt = "<DocVQA>where is the dinner?"
image = data[sample_idx]["image"].convert("RGB")

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)


generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<VQA>", image_size=(image.width, image.height))

# print(generated_text)
print(parsed_answer)

# %%
# HTR


from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = PROJECT_DIR / "models/florence-2-base-ft-hovratt-htr/"
remote_model_path = "microsoft/Florence-2-base-ft"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(remote_model_path, trust_remote_code=True)

hovratt_test = torch.load(PROJECT_DIR / "data/hovratt_test.pt", weights_only=False)

# %%
sample_idx = 500
prompt = "<HTR>Print out the text in this image"
image = hovratt_test[sample_idx]["image"].convert("RGB")
answer = hovratt_test[sample_idx]["transcription"]

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)


generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3,
)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_answer = processor.post_process_generation(generated_text, task="<HTR>", image_size=(image.width, image.height))

plt.imshow(image)
print("Mdl:\t", parsed_answer["<HTR>"])
print("Gt:\t", answer)

cer_metric = float(cer.compute(parsed_answer["<HTR>"], answer)["cer"])
wer_metric = float(wer.compute(parsed_answer["<HTR>"], answer)["wer"])

print(f"CER: {cer_metric:.4f}, WER: {wer_metric:.4f}")

# %%
