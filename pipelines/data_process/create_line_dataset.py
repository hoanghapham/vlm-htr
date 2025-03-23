#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from datasets import Dataset
from argparse import ArgumentParser

from src.data_process.running_text import RunningTextDatasetBuilder
from src.logger import CustomLogger


# Argparse
parser = ArgumentParser()
parser.add_argument("--input-dir", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()

# # Setup
INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)



# INPUT_DIRS = [
#     Path("/proj/uppmax2024-2-24/hapham/data/riksarkivet/bergskollegium_adv"),
#     Path("/proj/uppmax2024-2-24/hapham/data/riksarkivet/bergskollegium_rel"),
#     Path("/proj/uppmax2024-2-24/hapham/data/riksarkivet/frihetstidens"),
#     Path("/proj/uppmax2024-2-24/hapham/data/riksarkivet/jonkopings_radhusratts"),
#     Path("/proj/uppmax2024-2-24/hapham/data/riksarkivet/krigshovrattens"),
#     Path("/proj/uppmax2024-2-24/hapham/data/riksarkivet/svea_hovratt"),
#     Path("/proj/uppmax2024-2-24/hapham/data/riksarkivet/trolldoms"),
# ]

logger = CustomLogger("create_line_dataset")

#%%
builder = RunningTextDatasetBuilder()

img_ext = [".tif", ".jpg", ".jpeg", ".png"]

img_paths = [str(path) for path in sorted(Path.glob(INPUT_DIR / "images", pattern="**/*")) if path.suffix in img_ext]
xml_paths = [str(path) for path in sorted(Path.glob(INPUT_DIR / "page_xmls", pattern="**/*.xml"))]

# assert len(img_path) == len(all_xml_paths) > 0, \
#     f"Invalid length, or mismatch: {len(all_img_paths)} - {len(all_xml_paths)}"

imgs_xmls = list(zip(
        sorted(img_paths, key=lambda x: Path(x).stem), 
        sorted(xml_paths, key=lambda x: Path(x).stem)
    )
)

ttl_samples = len(img_paths)

for idx, (img_path, xml_path) in enumerate(imgs_xmls):
    logger.info(f"Process page {idx}/{ttl_samples}")

    page_name = Path(img_path).stem
        
    dataset_obj = Dataset.from_list(
        [
            {
                "unique_key": data["unique+key"],
                "image": data["image"],
                "transcription": data["transcription"]
            } for data in builder.create_line_dataset([(img_path, xml_path)])
        ]
    )

    dataset_obj.save_to_disk(OUTPUT_DIR / page_name)
    