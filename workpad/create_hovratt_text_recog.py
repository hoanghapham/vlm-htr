
#%%
from datasets import load_dataset
import time
import numpy as np
import torch

from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent

hovratt = load_dataset("Riksarkivet/gota_hovratt_seg", trust_remote_code=True, name="text_recognition")

# %%
total_samples = len(hovratt["train"])
np.random.seed(42)
train_idx = np.random.choice(range(total_samples), size=int(0.8 * total_samples), replace=False)
test_idx = np.array([i for i in range(total_samples) if i not in train_idx])

train_set = hovratt["train"].select(train_idx)
test_set = hovratt["train"].select(test_idx)

assert len(train_set) + len(test_set) == total_samples, "Mismatch subset sizes after splitting"
# %%

torch.save(train_set, PROJECT_DIR / "data/hovratt_train.pt")
torch.save(test_set, PROJECT_DIR / "data/hovratt_test.pt")
# %%
