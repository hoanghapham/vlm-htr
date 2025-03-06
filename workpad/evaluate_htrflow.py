#%%
from htrflow.evaluate import CER, WER, BagOfWords, read_xmls
from pathlib import Path
import sys
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from src.utils.file_tools import list_files

gt_dir = PROJECT_DIR / "data/poliskammare/page_xmls"
candidate_dir = PROJECT_DIR / "output/page"

# Read PageXML
print("Read ground truth")
ground_truth = read_xmls(gt_dir)


#%%
import os
from pagexml.parser import parse_pagexml_file
print("Read candidates")
# candidates = read_xmls(candidate_dir)


candidates = {}
for parent, _, files in os.walk(candidate_dir):
    for file in files:
        if not file.endswith(".xml"):
            continue
        try:
            page = parse_pagexml_file(os.path.join(parent, file))
        except Exception:
            print(file)
            continue
        candidates[file] = page


#%%
# Normalize
def NFD(s):
    return unicodedata.normalize('NFD', s)

ground_truth = {NFD(k): v for k, v in ground_truth.items()}
candidates = {NFD(k): v for k, v in candidates.items()}


#%%

pages = list(ground_truth.keys())
cer_metrics = []
wer_metrics = []
bow_metrics = []

metrics = [CER(), WER(), BagOfWords()]


pages = [page for page in ground_truth if page in candidates and ground_truth[page].num_words > 0]

print(len(page))


#%%
dfs = []
print("Calculate metrics")
for metric in metrics:
    values = [metric(ground_truth[page], candidates[page]) for page in pages]
    df = pd.DataFrame(values, index=pages)
    dfs.append(df)

df = pd.concat(dfs, axis=1)

for name in df.columns:
    df[name + "_float"] = df[name].apply(lambda x: float(x))

df.to_csv(PROJECT_DIR / "output/htrflow_metrics.csv", index=True)

# import numpy as np

# df.aggregate({
#     "cer": np.mean,
#     "wer": np.mean,
#     "bow_hits": np.mean,
#     "bow_extras": np.mean
# })


#%%

