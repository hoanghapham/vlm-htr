import sys
import shutil
import os
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser
from src.train import find_best_checkpoint, find_last_checkpoint

parser = ArgumentParser()
parser.add_argument("--models-dir", "-md", required=True)
parser.add_argument("--overwrite", "-ow", default="false")
args = parser.parse_args()

MODELS_DIR = Path(args.models_dir)
OVERWRITE = args.overwrite == "true"

model_paths = sorted([path for path in MODELS_DIR.iterdir() if path.is_dir()])

for model_path in model_paths:
    print(f"Copy best & last of {model_path.stem}")
    try:
        best_cp_path = find_best_checkpoint(model_path, "avg_val_loss")
        last_cp_path = find_last_checkpoint(model_path)

        best_cp_files = [path.name for path in best_cp_path.iterdir()]
        last_cp_files = [path.name for path in last_cp_path.iterdir()]

        # Best checkpoint
        assert "optimizer_state_dict.pt" in best_cp_files, f"optimizer_sate_dict.pt not in {best_cp_path.stem}"
        assert "model.safetensors" in best_cp_files, f"model.safetensors not in {best_cp_path.stem}"

        if (best_cp_path.parent / "best").exists():
            if OVERWRITE:
                shutil.copytree(best_cp_path, best_cp_path.parent / "best", dirs_exist_ok=True)
            else:
                print("best checkpoint exists, skip")
        else:
            shutil.copytree(best_cp_path, best_cp_path.parent / "best")

        # Last checkpoint
        assert "optimizer_state_dict.pt" in last_cp_files, f"optimizer_sate_dict.pt not in {last_cp_path.stem}"
        assert "model.safetensors" in last_cp_files, f"model.safetensors not in {last_cp_path.stem}"
        shutil.copytree(last_cp_path, last_cp_path.parent / "last", dirs_exist_ok=True)

        # Delete optimizer states
        opt_state_paths = [path for path in model_path.glob("checkpoint_*/optimizer_state_dict.pt")]

        counter = 0
        for path in opt_state_paths:
            os.remove(path)
            counter += 1
        
        print(f"Delete {counter} optimizer states.")

        # Delete model states
        counter = 0
        model_state_paths = [path for path in model_path.glob("checkpoint_*/model.safetensors")]
        for path in model_state_paths:
            os.remove(path)
            counter += 1

        print(f"Delete {counter} model states.")

    except Exception as e:
        print(e)
        continue


