import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser

import torch
from torch.optim import AdamW
from transformers import (
    VisionEncoderDecoderModel, 
    AutoModelForCausalLM, 
    TrOCRProcessor

)
from peft import get_peft_model, LoraConfig
from src.file_tools import read_json_file
from src.train import save_checkpoint


parser = ArgumentParser()
parser.add_argument(nargs='+', dest="model_dirs")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dirs = [Path(model_dir) for model_dir in args.model_dirs]

for model_dir in model_dirs:

    print(f"Model: {model_dir.name}")

    if "lora" in model_dir.name:
        print("skip")
        continue

    if "florence" in model_dir.name:
        REMOTE_MODEL_PATH = "microsoft/Florence-2-base-ft"
        REVISION = 'refs/pr/6'
        model = AutoModelForCausalLM.from_pretrained(
            REMOTE_MODEL_PATH,
            trust_remote_code=True,
            revision=REVISION
        ).to(DEVICE)

        # if "lora" in model_dir.name:
        #     config = LoraConfig.from_pretrained(PROJECT_DIR / "configs/lora")
        #     model = get_peft_model(model, config)

        optimizer = AdamW(model.parameters(), lr=1e-6)

    elif "trocr" in model_dir.name:
        REMOTE_MODEL_PATH = "microsoft/trocr-base-handwritten"
        processor   = TrOCRProcessor.from_pretrained(REMOTE_MODEL_PATH)
        model       = VisionEncoderDecoderModel.from_pretrained(REMOTE_MODEL_PATH).to(DEVICE)
        optimizer   = AdamW(model.parameters(), lr=2e-5, weight_decay=0.0001)
        
        model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
        model.config.pad_token_id           = processor.tokenizer.pad_token_id
        model.config.vocab_size             = model.config.decoder.vocab_size

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