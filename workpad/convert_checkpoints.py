import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from argparse import ArgumentParser
from tqdm import tqdm

from src.file_tools import write_json_file

parser = ArgumentParser()
parser.add_argument("--model-dir")
args = parser.parse_args()

MODEL_DIR = Path(args.model_dir)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cp_state_paths = sorted(MODEL_DIR.glob("*.pt"))

for path in tqdm(cp_state_paths, total=len(cp_state_paths)):
    info = torch.load(path, weights_only=True, map_location=torch.device(DEVICE))

    cp_metrics = dict(epoch=info["epoch"], loss=info["loss"].item())
    cp_state = dict(model_state_dict=info["model_state_dict"], optimizer_state_dict=info["optimizer_state_dict"])

    epoch = info["epoch"]

    write_json_file(cp_metrics, MODEL_DIR / f"checkpoint_epoch_{epoch:04d}.json")
    torch.save(cp_state, MODEL_DIR / f"checkpoint_epoch_{epoch:04d}.pt")
