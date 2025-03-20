#%%
import sys
import yaml
import time
from pathlib import Path
from argparse import ArgumentParser

from htrflow.pipeline.pipeline import Pipeline
from htrflow.volume.volume import Collection

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import list_files, read_json_file
from src.logger import CustomLogger

parser = ArgumentParser()
parser.add_argument("--input-dir", required=True)
parser.add_argument("--split-info-fp", required=False)
parser.add_argument("--config-path", required=True)
parser.add_argument("--img-extension", required=True, default=".tif")
args = parser.parse_args()

logger = CustomLogger("HTRFlow", log_to_local=True)

# Parse args
INPUT_DIR       = Path(args.input_dir)
CONFIG_PATH     = Path(args.config_path)


#%%
# Load the YAML configuration.
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Load split info
split_info = None

if args.split_info_fp:
    split_info = read_json_file(args.split_info_fp)

#%%
# Create a pipeline instance from the loaded configuration.
pipe = Pipeline.from_config(config)

# Create collection
image_files = list_files(INPUT_DIR, extensions=[args.img_extension])

if split_info:
    image_files = [tup for tup in image_files if tup[1].stem in split_info["test"]]

# existing_xml_files  = [name for (_, name) in list_files(PROJECT_DIR / "output", extensions=[".xml"])]

logger.info(f"Total images to be processed: {len(image_files)}")

# Iterate through n images at a time

batch_size = 100
total_images = len(image_files)

for i in range((total_images // batch_size) + 1):
    start = i * batch_size
    end = min((i+1) * batch_size, total_images)

    images = [
        str(parent / file) for parent, file in image_files[start:end] 
        # if str(Path(file).with_suffix(".xml")) not in existing_xml_files
    ]

    logger.info(f"Process images {start} - {end}")
    
    if len(images) > 0:

        t0 = time.time()
        collection = Collection(images)
        logger.info(f"Create collection: {(time.time()-t0) / 60:.2f} minutes")

        # logger.info("Run pipeline")
        t0 = time.time()
        new_collection = pipe.run(collection)
        logger.info(f"Inference time: {(time.time()-t0) / 60:.2f} minutes")


