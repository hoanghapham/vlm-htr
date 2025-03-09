#%%
import sys
from pathlib import Path

from datasets import Dataset
import json
from dotenv import dotenv_values

PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from src.image_tools.poliskammare import ImageDatasetBuilder
from src.utils import gen_split_indices
from src.logger import CustomLogger


# Setup
env_dict = dotenv_values(PROJECT_DIR / ".env")
DATA_DIR = Path(env_dict["POLIS_DATA_DIR"])
OUTPUT_DIR = PROJECT_DIR / "data/poliskammare_region"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

logger = CustomLogger("create_region_dataset")

#%%
builder = ImageDatasetBuilder()

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

split_info = {
    "train": [all_img_paths[idx] for idx in train_indices],
    "validation": [all_img_paths[idx] for idx in val_indices],
    "test": [all_img_paths[idx] for idx in test_indices]
}


with open(OUTPUT_DIR / "split_info.json", "w") as f:
    json.dump(split_info, f)


subsets = [
    ("train", [imgs_xmls[idx] for idx in train_indices]),
    ("validation", [imgs_xmls[idx] for idx in val_indices]),
    ("test", [imgs_xmls[idx] for idx in test_indices])
]

for subset_name, pairs in subsets:

    logger.info(f"Process {subset_name}")

    dataset_obj = Dataset.from_list(
        [
            {
                "id": data[0],
                "image": data[1]["image"],
                "transcription": data[1]["transcription"]
            } for data in builder.create_region_dataset(pairs)
        ]   
    )

    dataset_obj.save_to_disk(OUTPUT_DIR / subset_name)
    
