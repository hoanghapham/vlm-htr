#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from datasets import Dataset

from src.data_processing.visual_tasks import HTRDatasetBuilder
from src.logger import CustomLogger
from src.file_tools import list_files

# Setup
DATA_DIR    = PROJECT_DIR / "data/raw/riksarkivet"
OUTPUT_DIR  = PROJECT_DIR / "data/processed/riksarkivet_region"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

logger = CustomLogger("create_region_dataset")

#%%
builder = HTRDatasetBuilder(config_name="text_recognition")

for dir_path in DATA_DIR.iterdir():
    
    if dir_path.is_dir():
        subset_name = dir_path.stem
        logger.info(f"Create region dataset for {subset_name}")

        img_paths = list_files(dir, [".tif", ".jpg"])
        xml_paths = list_files(dir, [".xml"])
        matches = set([path.stem for path in img_paths]).intersection(set([path.stem for path in xml_paths]))
        assert len(img_paths) == len(xml_paths) == len(matches) > 0, \
            f"Invalid length, or mismatch: {len(img_paths)} - {len(xml_paths)} - {len(matches)}"
        
        imgs_xmls = list(zip(
                sorted(img_paths, key=lambda x: Path(x).stem), 
                sorted(xml_paths, key=lambda x: Path(x).stem)
            )
        )

        for idx, (img_path, xml_path) in enumerate(imgs_xmls):

            logger.info(f"Process image {idx}/{len(imgs_xmls)}")
            file_name = Path(img_path).stem

            dataset_obj = Dataset.from_list(
                [
                    {
                        "id": data[0],
                        "image": data[1]["image"],
                        "transcription": data[1]["transcription"]
                    } for data in builder.create_smooth_region_dataset([(img_path, xml_path)])
                ]   
            )

            dataset_obj.save_to_disk(OUTPUT_DIR / file_name)
    