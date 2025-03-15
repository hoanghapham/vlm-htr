#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser
from datasets import Dataset

from src.image_tools.poliskammare import ImageDatasetBuilder
from src.logger import CustomLogger


parser = ArgumentParser()
parser.add_argument("--input-dir", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()

# Setup
INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

logger = CustomLogger("create_region_dataset")

#%%
logger.info(f"Create region dataset for {INPUT_DIR}")
builder = ImageDatasetBuilder()

all_img_paths = [str(path) for path in sorted(Path.glob(INPUT_DIR / "images", pattern="**/*.tif"))]
all_xml_paths = [str(path) for path in sorted(Path.glob(INPUT_DIR / "page_xmls", pattern="**/*.xml"))]

assert len(all_img_paths) == len(all_xml_paths) > 0, \
    f"Invalid length, or mismatch: {len(all_img_paths)} - {len(all_xml_paths)}"

imgs_xmls = list(zip(
        sorted(all_img_paths, key=lambda x: Path(x).stem), 
        sorted(all_xml_paths, key=lambda x: Path(x).stem)
    )
)

ttl_samples = len(all_img_paths)

for idx, (img_path, xml_path) in enumerate(imgs_xmls):

    logger.info(f"Process image {idx}/{ttl_samples}")
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
    