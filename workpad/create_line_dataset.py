#%%
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from collections import defaultdict
from tqdm import tqdm
import json
from dotenv import dotenv_values

PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from src.image_tools.poliskammare import LineDataset
from src.utils import gen_split_indices


# Setup
env_dict = dotenv_values(PROJECT_DIR / ".env")
DATA_DIR = Path(env_dict["POLIS_DATA_DIR"])
OUTPUT_DIR = PROJECT_DIR / "data/poliskammare_line"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

#%%
line_dataset = LineDataset(name="line")

all_img_paths = [str(path) for path in sorted(Path.glob(DATA_DIR / "images", pattern="**/*.tif"))]
all_xml_paths = [str(path) for path in sorted(Path.glob(DATA_DIR / "page_xmls", pattern="**/*.xml"))]

assert len(all_img_paths) == len(all_xml_paths) > 0, \
    f"Invalid length, or mismatch: {len(all_img_paths)} - {len(all_xml_paths)}"

imgs_xmls = list(zip(
        sorted(all_img_paths, key=lambda x: Path(x).stem), 
        sorted(all_xml_paths, key=lambda x: Path(x).stem)
    )
)

ttl_samples = len(all_img_paths)
train_indices, val_indices, test_indices = gen_split_indices(ttl_samples, seed=42)

#%%

subsets = [
    ("train", train_indices), 
    ("validation", val_indices),
    ("test", test_indices)
]

splits = {}

for subset_name, indices in subsets:
    print(f"Process {subset_name} set:")
    imgs_xmls_subset = [imgs_xmls[idx] for idx in indices]
    splits[subset_name] = imgs_xmls_subset
        
    dataset_obj = Dataset.from_list(
        [
            {
                "id": data[0],
                "image": data[1]["image"],
                "transcription": data[1]["transcription"]
            } for data in line_dataset.text_recognition(imgs_xmls_subset)
        ]
    )

    dataset_obj.save_to_disk(OUTPUT_DIR / subset_name)
    

with open(OUTPUT_DIR / "split_info.json", "w") as f:
    json.dump(splits, f)