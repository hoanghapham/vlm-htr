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

from src.file_tools import read_json_file
from src.train import save_checkpoint, STEP_IDX_SPACES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dirs = (PROJECT_DIR / "models").iterdir()

for model_dir in models_dirs:
    print(f"Model: {model_dir.name}")

    if "florence" in model_dir.name:
        REMOTE_MODEL_PATH = "microsoft/Florence-2-base-ft"
        REVISION = 'refs/pr/6'
        model = AutoModelForCausalLM.from_pretrained(
            REMOTE_MODEL_PATH,
            trust_remote_code=True,
            revision=REVISION
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=1e-6)

    elif "troc" in model_dir.name:
        REMOTE_MODEL_PATH = "microsoft/trocr-base-handwritten"
        model = VisionEncoderDecoderModel.from_pretrained(REMOTE_MODEL_PATH).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.0001)

    else:
        continue

    step_idx = int(model_dir.name.split("_")[-1])
    json_files  = sorted(model_dir.glob("*.json"))
    pt_files    = sorted(model_dir.glob("*.pt"))

    for json, pt in zip(json_files, pt_files):
        
        metrics = read_json_file(json)
        states = torch.load(pt, map_location=DEVICE)

        model.load_state_dict(states["model_state_dict"])
        optimizer.load_state_dict(states["optimizer_state_dict"])

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            out_dir=model_dir / f"checkpoint_step_{str(step_idx).zfill(STEP_IDX_SPACES)}",
            metrics=metrics
        )

        print(f"Processed {json.stem}")

        



