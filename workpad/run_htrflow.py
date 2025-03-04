#%%
from htrflow.pipeline.pipeline import Pipeline
from htrflow.volume.volume import Collection
from htrflow.pipeline.steps import auto_import
import yaml
from pathlib import Path
from argparse import ArgumentParser
import sys
import time

PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from src.utils.file_tools import list_files
from src.utils.logger import CustomLogger

parser = ArgumentParser()
parser.add_argument("--input_dir", "-i", required=True)
parser.add_argument("--config_path", "-c", required=True)
args = parser.parse_args()

logger = CustomLogger("HTRFlow", log_to_local=True)

# Parse args
input_dir = Path(args.input_dir)
config_path = Path(args.config_path)

#%%
# Load the YAML configuration.
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

#%%
# Create a pipeline instance from the loaded configuration.
pipe = Pipeline.from_config(config)

# Create collection
image_files = list_files(input_dir, extensions=[".tif"])
parents: list[Path] = list(set([tup[0] for tup in image_files]))
existing_xml_files = [name for (_, name) in list_files(PROJECT_DIR / "output", extensions=[".xml"])]

logger.info(f"Found {len(parents)} parent dirs, {len(image_files)} images")

# Iterate through n images at a time

batch_size = 100
total_images = len(image_files)

for i in range(total_images // batch_size):
    start = i * batch_size
    end = min((i+1) * batch_size, total_images)

    images = [
        str(parent / file) for parent, file in image_files[start:end] 
        if str(Path(file).with_suffix(".xml")) not in existing_xml_files
    ]

    logger.info(f"Process images {start} - {end}")
    
    if len(images) > 0:

        logger.info("Create collection")
        t0 = time.time()
        collection = Collection(images)
        logger.info(f"Create collection: {(time.time()-t0) / 60} minutes")

        logger.info("Run pipeline")
        t0 = time.time()
        new_collection = pipe.run(collection)
        logger.info(f"Inference time: {(time.time()-t0) / 60} minutes")


