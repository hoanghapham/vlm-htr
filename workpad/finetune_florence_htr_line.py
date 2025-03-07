
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent

# Load model
print("Load model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "microsoft/Florence-2-base-ft"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    revision='refs/pr/6'
).to(device)

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True, revision='refs/pr/6'
)

# Unfreeze vision params
for param in model.vision_tower.parameters():
  param.is_trainable = True


# Create VQA Dataset
# Load data
print("Load data")
data = load_dataset("Riksarkivet/gota_hovratt_seg", trust_remote_code=True, name="text_recognition")

total_samples = len(data["train"])
np.random.seed(42)
train_idx = np.random.choice(range(total_samples), size=int(0.8 * total_samples), replace=False)
val_idx = np.array([i for i in range(total_samples) if i not in train_idx])

assert len(train_idx) + len(val_idx) == total_samples, "Mismatch subset sizes after splitting"

class HTRDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<HTR>Print out the text in this image"
        answer = example["transcription"]
        image = example['image'].convert("RGB")
        return question, answer, image


# Create train loader & validate loader
# Processor comes from when loading the model
def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers



# Subset train & validate set
train_dataset = HTRDataset(data['train'].select(train_idx))
val_dataset = HTRDataset(data['train'].select(val_idx))

batch_size = 2
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          collate_fn=collate_fn, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          collate_fn=collate_fn, num_workers=num_workers)


# Train
epochs = 5
optimizer = AdamW(model.parameters(), lr=1e-6)
num_training_steps = epochs * len(train_loader)


lr_scheduler = get_scheduler(name="linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps,)


for epoch in range(epochs):
    model.train()
    train_loss = 0

    # Inputs is the processed tuple (text, image)
    for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):

        # Get input (text tokens and )
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)

        # Predict output
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)

        # Calculate loss, then backward
        loss = outputs.loss
        loss.backward()

        # Then step
        optimizer.step()
        lr_scheduler.step()

        # Reset grad
        optimizer.zero_grad()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss}")


# Check validation loss
model.eval()
val_loss = 0
with torch.no_grad():
    for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
        inputs, answers = batch
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        val_loss += loss.item()

print("Average Validation Loss: ", val_loss / len(val_loader))



# Save model
print("Save model")
model_out_dir = PROJECT_DIR / "models/florence-2-base-ft-hovratt-htr-line"

if not model_out_dir.exists():
    model_out_dir.mkdir(parents=True)

model.save_pretrained(model_out_dir, from_pt=True)