#%%
from htrflow.evaluate import CER, WER, BagOfWords, read_xmls
from pathlib import Path
import sys
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from src.utils.file_tools import list_files

gt_dir = PROJECT_DIR / "data/poliskammare/page_xmls"
candidate_dir = PROJECT_DIR / "workpad/outputs/page"

# Read PageXML
ground_truth = read_xmls(gt_dir)
candidates = read_xmls(candidate_dir)

pages = list(ground_truth.keys())
cer_metrics = []
wer_metrics = []
bow_metrics = []

metrics = [CER(), WER(), BagOfWords()]

dfs = []
pages = [page for page in ground_truth if page in candidates and ground_truth[page].num_words > 0]

for metric in metrics:
    values = [metric(ground_truth[page], candidates[page]) for page in pages]
    df = pd.DataFrame(values, index=pages)
    dfs.append(df)

df = pd.concat(dfs, axis=1)
# %%
