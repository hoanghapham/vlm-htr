#%%
from htrflow.evaluate import CER, WER, BagOfWords, LineCoverage, RegionCoverage, read_xmls
from pagexml.parser import parse_pagexml_file
from pathlib import Path
import pandas as pd
import unicodedata
from tqdm import tqdm

from vlm.utils.file_tools import list_files,


task = "hovratt_text_recognition"

PROJECT_DIR = Path(__file__).parent.parent.parent
groundtruth_dir = PROJECT_DIR / "data/hovratt/page_xmls"
candidate_dir   = PROJECT_DIR / f"output/htrflow/{task}/page"


split_info = None
# split_info = read_json_file(PROJECT_DIR / "data/polis_line/split_info.json")


def NFD(s):
    return unicodedata.normalize('NFD', s)

def read_xmls(dir_path: str | Path):
    xml_files = list_files(dir_path, extensions=[".xml"])
    results = {}
    error_files = []

    for parent, file_name in tqdm(xml_files, unit="file"):
        try:
            page = parse_pagexml_file(parent / file_name)
        except Exception:
            error_files.append(file_name)
            continue
        results[file_name] = page
    if error_files:
        print(f"Fail to parse {len(error_files)} files")
    return results
#%%

# Read PageXML
print("Read xmls")
candidates = read_xmls(candidate_dir)
groundtruth = read_xmls(groundtruth_dir)


#%%
# Normalize

groundtruth = {NFD(k): v for k, v in groundtruth.items()}
candidates = {NFD(k): v for k, v in candidates.items()}

if split_info:
    train_pages = [NFD(name) for name in split_info["train"]]
    validation_pages = [NFD(name) for name in split_info["validation"]]
    test_pages = [NFD(name) for name in split_info["test"]]

# Filter pages
pages = list(groundtruth.keys())
cer_metrics = []
wer_metrics = []
bow_metrics = []

pages = [
    page for page in groundtruth 
    if page in candidates 
        and groundtruth[page].num_words > 0
        # and page.replace(".xml", "") in test_pages
    ]



print(f"Evaluate performance on {len(pages)} pages")


#%%

metrics = [LineCoverage(), RegionCoverage(), CER(), WER(), BagOfWords()]

processed_pages = []
dfs = []

print("Calculate metrics")
for metric in metrics:
    values = []
    for page in tqdm(pages):
        values.append(metric(groundtruth[page], candidates[page]))

    df = pd.DataFrame(values, index=pages)
    dfs.append(df)


#%%

result_df = pd.concat(dfs, axis=1)
for name in result_df.columns:
    result_df[name + "_float"] = result_df[name].apply(lambda x: float(x) if x else pd.NA)

# Write result
result_df = result_df.reset_index().rename(columns={"index": "page"})
result_df.to_csv(PROJECT_DIR / f"output/htrflow/{task}_metrics.csv", index=False)



# %%
