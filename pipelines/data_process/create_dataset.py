#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser
from datasets import Dataset

from src.data_processing.visual_tasks import HTRDataset
from src.logger import CustomLogger
from src.file_tools import list_files


parser = ArgumentParser()
parser.add_argument("--raw-data-dir", required=True)
parser.add_argument("--dataset-type", default="text_recognition")
parser.add_argument("--processed-data-dir", default="data/processed")
args = parser.parse_args()


# Setup
RAW_DATA_DIR        = Path(args.raw_data_dir)
PROCESSED_DATA_DIR  = Path(args.processed_data_dir)
DATASET_TYPE        = args.dataset_type
OUTPUT_DIR          = PROCESSED_DATA_DIR / DATASET_TYPE


if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)


logger = CustomLogger(f"create_{DATASET_TYPE}")

#%%

image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
            ".JPG",
            ".JPEG",
            ".PNG",
            ".GIF",
            ".BMP",
            ".TIF",
            ".TIFF",]

builder = HTRDataset(config_name=DATASET_TYPE)

process_funcs = {
    "text_recognition": builder.text_recognition,   # cropped lines & transcription
    "inst_seg_lines": builder.inst_seg_lines,       # Original images with text line annotations only
    "inst_seg_regions": builder.inst_seg_regions,   # Original images with text region annotations only

    # Cropped text region images with text line annotations
    "inst_seg_lines_within_regions": builder.inst_seg_lines_within_regions, 
    
    # Original images with both region and line annotations, can be used for full object detection 
    "inst_seg_regions_and_lines": builder.inst_seg_regions_and_lines,       
}


logger.info(f"Create dataset: {DATASET_TYPE}")

for dir_path in RAW_DATA_DIR.iterdir():
    
    if dir_path.is_dir():
        subset_name = dir_path.stem
        logger.info(f"Process {subset_name}")

        img_paths = list_files(dir_path, image_extensions)
        xml_paths = list_files(dir_path, [".xml"])
        matches = set([path.stem for path in img_paths]).intersection(set([path.stem for path in xml_paths]))
        assert len(img_paths) == len(xml_paths) == len(matches) > 0, \
            f"Invalid length, or mismatch: {len(img_paths)} - {len(xml_paths)} - {len(matches)}"
        
        imgs_xmls = list(zip(
                sorted(img_paths, key=lambda x: Path(x).stem), 
                sorted(xml_paths, key=lambda x: Path(x).stem)
            )
        )

        for idx, img_xml in enumerate(imgs_xmls):

            logger.info(f"Process image {idx}/{len(imgs_xmls)}")
            file_name = Path(img_xml[0]).stem

            data_list = list(process_funcs[DATASET_TYPE]([img_xml]))

            if len(data_list) == 0:
                continue

            data = Dataset.from_list(data_list)
            data.save_to_disk(OUTPUT_DIR / file_name)
    