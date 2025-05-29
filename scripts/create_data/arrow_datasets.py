# Script to create arrow datasets that is caches locally. Possible usages: datasets of cropped lines, cropped regions etc...
#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from argparse import ArgumentParser
from datasets import Dataset

from src.data_processing.visual_tasks import HTRDatasetBuilder, IMAGE_EXTENSIONS
from src.logger import CustomLogger
from src.file_tools import list_files, write_json_file


dataset_types = [
    "text_recognition__line_seg",
    "text_recognition__line_bbox",
    "inst_seg_lines",
    "inst_seg_regions",
    "inst_seg_lines_within_regions",
    "line_od_within_regions",
    "inst_seg_regions_and_lines",
]

parser = ArgumentParser()
parser.add_argument("--raw-data-dir", required=True)
parser.add_argument("--dataset-type", default="text_recognition__line_seg", choices=dataset_types)
parser.add_argument("--output-data-dir", required=True)
parser.add_argument("--overwrite", default="false")
args = parser.parse_args()


# Setup
RAW_DATA_DIR        = Path(args.raw_data_dir)
DATASET_TYPE        = args.dataset_type
OUTPUT_DATA_DIR     = Path(args.output_data_dir)
OVERWRITE           = args.overwrite == "true"

if not OUTPUT_DATA_DIR.exists():
    OUTPUT_DATA_DIR.mkdir(parents=True)


logger = CustomLogger(f"create_{DATASET_TYPE}")

#%%

builder = HTRDatasetBuilder(config_name=DATASET_TYPE)

process_funcs = {
    # cropped lines (polygon) & transcription
    "text_recognition__line_seg": builder.text_recognition__line_seg,       

    # cropped lines (bbox) & transcription
    "text_recognition__line_bbox": builder.text_recognition__line_bbox,     

    # Original images with text line annotations only
    "inst_seg_lines": builder.inst_seg_lines,       

    # Original images with text region annotations only
    "inst_seg_regions": builder.inst_seg_regions,   

    # Cropped text region images with text line segmentation
    "inst_seg_lines_within_regions": builder.inst_seg_lines_within_regions, 
    
    # Cropped text region images with text line detection
    "line_od_within_regions": builder.line_od_within_regions, 
    
    # Original images with both region and line annotations, can be used for full object detection 
    "inst_seg_regions_and_lines": builder.inst_seg_regions_and_lines,       
}


logger.info(f"Create dataset: {DATASET_TYPE}")

processed_pages = set([file.parent.stem for file in list_files(OUTPUT_DATA_DIR, [".arrow"])])

subset_stats = {}

    
subset_name = RAW_DATA_DIR.stem
logger.info(f"Process {subset_name}")

img_paths = list_files(RAW_DATA_DIR, IMAGE_EXTENSIONS)
xml_paths = list_files(RAW_DATA_DIR, [".xml"])
matches = set([path.stem for path in img_paths]).intersection(set([path.stem for path in xml_paths]))
assert len(img_paths) == len(xml_paths) == len(matches) > 0, \
    f"Invalid length, or mismatch: {len(img_paths)} - {len(xml_paths)} - {len(matches)}"

imgs_xmls = list(zip(
        sorted(img_paths, key=lambda x: Path(x).stem), 
        sorted(xml_paths, key=lambda x: Path(x).stem)
    )
)

subset_stats[subset_name] = {"total": len(imgs_xmls)}
failed = 0
no_data = 0

failed_files = []
no_data_files = []

for idx, img_xml in enumerate(imgs_xmls):
    file_name = Path(img_xml[0]).stem
    
    if not OVERWRITE:
        if file_name in processed_pages:
            logger.info(f"Already processed, skip: {file_name}")
            continue
    
    logger.info(f"Subset: {subset_name}, image {idx}/{len(imgs_xmls)}")
    try:
        # Process data using the process funct
        data_list = list(process_funcs[DATASET_TYPE]([img_xml]))
    except Exception as e:
        logger.warning(f"Image {file_name} failed: {e}")
        failed += 1
        failed_files.append(file_name)
        continue

    if len(data_list) == 0:
        logger.warning(f"No data to write: {img_xml[0]}")
        no_data += 1
        no_data_files.append(file_name)
        continue
    
    # logger.info(f"Write data to {OUTPUT_DATA_DIR / file_name}")
    data = Dataset.from_list(data_list)
    data.save_to_disk(OUTPUT_DATA_DIR / file_name)

subset_stats[subset_name]["failed"] = failed
subset_stats[subset_name]["failed_files"] = failed_files
subset_stats[subset_name]["no_data"] = no_data
subset_stats[subset_name]["no_data_files"] = no_data_files
        

write_json_file(subset_stats, f"temp/{subset_name}_stats.json")