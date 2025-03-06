
#%%
from datasets import load_dataset
import time
polis = load_dataset("Riksarkivet/goteborgs_poliskammare_fore_1900", trust_remote_code=True, name="text_recognition")

# %%
