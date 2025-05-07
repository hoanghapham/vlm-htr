import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from htrflow.evaluate import Ratio
from src.file_tools import read_json_file

def read_img_metrics(
    img_metric_path: str | Path, 
    cer_list: list, 
    wer_list: list, 
    bow_hits_list: list, 
    bow_extras_list: list
) -> tuple[list, list, list, list]:
    img_metric = read_json_file(img_metric_path)
    cer_list.append(Ratio(*img_metric["cer"]["str"].split("/")))
    wer_list.append(Ratio(*img_metric["wer"]["str"].split("/")))
    bow_hits_list.append(Ratio(*img_metric["bow_hits"]["str"].split("/")))
    bow_extras_list.append(Ratio(*img_metric["bow_extras"]["str"].split("/")))

    return cer_list, wer_list, bow_hits_list, bow_extras_list
