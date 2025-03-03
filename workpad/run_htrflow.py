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
images = [str(parent / file) for (parent, file) in image_files]

collection = Collection(images)

# Run pipeline
logger.info("Start pipeline")
new_collection = pipe.run(collection)

logger.info("End pipeline")