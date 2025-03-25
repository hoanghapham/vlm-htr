#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser
from src.data_process.utils import gen_split_indices
from src.file_tools import read_json_file, write_json_file

#%%

parser = ArgumentParser()
parser.add_argument("--data-dir", required=True)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)

split_info_path = DATA_DIR / "split_info.json"


def create_split_info(
    data_dir: Path | str,
    seed: int = 42,
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15
):
    img_ext = [".tif", ".jpg", ".jpeg", ".png"]
    data_dir = Path(data_dir)
    split_info_path = Path(data_dir) / "split_info.json"
    img_paths = [path for path in sorted(data_dir.glob("**/images/**/*")) if path.is_file() and path.suffix in img_ext]
    xml_paths = [path for path in sorted(data_dir.glob("**/page_xmls/**/*.xml")) if path.is_file()]

    assert len(img_paths) == len(xml_paths) > 0, f"Length invalid: {len(img_paths)} images, {len(xml_paths)} xmls."

    train_indices, val_indices, test_indices = gen_split_indices(
        len(img_paths), 
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    split_info = {
        "train": [(str(img_paths[idx].absolute()), str(xml_paths[idx].absolute())) for idx in train_indices],
        "val": [(str(img_paths[idx].absolute()), str(xml_paths[idx].absolute())) for idx in val_indices],
        "test": [(str(img_paths[idx].absolute()), str(xml_paths[idx].absolute())) for idx in test_indices]
    }
    write_json_file(split_info, split_info_path)


#%%
create_split_info(DATA_DIR, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
split_info = read_json_file(split_info_path)

print({key: len(value) for key, value in split_info.items()})
# %%
