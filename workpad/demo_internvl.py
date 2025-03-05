#%%
import torch
from transformers import AutoTokenizer, AutoModel

model_path = "OpenGVLab/InternVL2-1B"
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True
).eval()
# %%
