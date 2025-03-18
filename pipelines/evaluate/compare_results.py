#%%
import pandas as pd
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.file_tools import read_json_file

#%%
MODEL_DIR = PROJECT_DIR / "output/florence-2-base-ft-htr-region"

metric_fps = sorted(MODEL_DIR.glob("**/metrics_aggr.json"))

all_tasks = []

for fp in metric_fps:
    data = dict(
        model = fp.parent.parts[-2],
        task = fp.parent.parts[-1]
    )
    metrics = read_json_file(fp)
    data.update(metrics)
    all_tasks.append(data)


pd.DataFrame(all_tasks)

# %%
