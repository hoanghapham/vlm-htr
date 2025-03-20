#%%
import torch
from ultralytics import YOLO
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

#%%

# Load the trained YOLOv9 model
# model = YOLO(PROJECT_DIR / 'models/yolov9__htr_region/model.pt')  # Update with the correct path to your trained model

model = YOLO("yolov9c.yaml")

#%%

DATA_DIR = Path("/Users/hoanghapham/Projects/thesis-data")

# Load an image
img = DATA_DIR / "poliskammare/images/Göteborgs_poliskammare_före_1900__A_II__1a__1868_/Göteborgs_poliskammare_före_1900__A_II__1a__1868__30002021_00011.tif"

# Perform instance segmentation
results = model(img)

# Display results
#%%
results[0].show()  # Show image with predicted masks

# To get the raw predictions (bounding boxes, masks, etc.)
for result in results:
    print(result.boxes)  # Bounding boxes
    print(result.masks)  # Segmentation masks

# %%
