import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

import torch
from torch.optim import AdamW
from transformers import (
    VisionEncoderDecoderModel, 
    AutoModelForCausalLM, 

)
from peft import get_peft_model, LoraConfig
from src.file_tools import read_json_file
from src.train import save_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dirs = (PROJECT_DIR / "models").iterdir()

for model_dir in model_dirs:

    print(f"Model: {model_dir.name}")

    if "florence" in model_dir.name:
        REMOTE_MODEL_PATH = "microsoft/Florence-2-base-ft"
        REVISION = 'refs/pr/6'
        model = AutoModelForCausalLM.from_pretrained(
            REMOTE_MODEL_PATH,
            trust_remote_code=True,
            revision=REVISION
        ).to(DEVICE)

        if "lora" in model_dir.name:
            config = LoraConfig.from_pretrained(PROJECT_DIR / "configs/lora")
            model = get_peft_model(model, config)

        optimizer = AdamW(model.parameters(), lr=1e-6)

    elif "trocr" in model_dir.name:
        REMOTE_MODEL_PATH = "microsoft/trocr-base-handwritten"
        model = VisionEncoderDecoderModel.from_pretrained(REMOTE_MODEL_PATH).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.0001)

    else:
        continue

    json_files  = sorted(model_dir.glob("*.json"))
    pt_files    = sorted(model_dir.glob("*.pt"))

    for json, pt in zip(json_files, pt_files):
        print(f"Process {json.stem}")
        
        metrics = read_json_file(json)
        states = torch.load(pt, map_location=DEVICE)

        print("Load model state dict")
        model.load_state_dict(states["model_state_dict"])

        print("Load optimizer state dict")
        optimizer.load_state_dict(states["optimizer_state_dict"])

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            out_dir=model_dir / json.stem,
            metrics=metrics
        )
        print("Saved")