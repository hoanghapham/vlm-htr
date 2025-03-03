#%%
from htrflow.pipeline.pipeline import Pipeline
from htrflow.volume.volume import Collection
from htrflow.pipeline.steps import auto_import
import yaml
from pathlib import Path
from argparse import ArgumentParser
import sys

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

logger.info(f"Found {len(parents)} parent dirs")

# Iterate through parents and list images
for parent in parents:
    images = [str(path) for path in parent.glob("**/*.tif")]
    logger.info(f"Folder {parent.stem}: {len(images)} images")
    
    if len(images) > 0:

        logger.info("Create collection")
        collection = Collection(images)

        logger.info("Run pipeline")
        new_collection = pipe.run(collection)

