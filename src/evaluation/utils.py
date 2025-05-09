
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from src.data_processing.utils import XMLParser
from src.evaluation.ocr_metrics import compute_ocr_metrics, OCRMetrics
from src.file_tools import write_text_file, write_json_file, read_json_file
from src.data_types import Page, Ratio


def read_metric_dict(metric_dict_path: str | Path, ) -> OCRMetrics:
    metric_dict  = read_json_file(metric_dict_path)
    cer         = Ratio(*metric_dict["cer"]["str"].split("/"))
    wer         = Ratio(*metric_dict["wer"]["str"].split("/"))
    bow_hits    = Ratio(*metric_dict["bow_hits"]["str"].split("/"))
    bow_extras  = Ratio(*metric_dict["bow_extras"]["str"].split("/"))
    return OCRMetrics(cer, wer, bow_hits, bow_extras)


def evaluate_one_page(page_obj: Page, gt_xml_path: Path, output_dir: Path = None):
    xml_parser = XMLParser()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    name = Path(gt_xml_path).stem

    gt_lines    = xml_parser.get_lines(gt_xml_path)
    gt_text     = " ".join([line["transcription"] for line in gt_lines])

    # Evaluation
    try:
        page_metrics = compute_ocr_metrics(page_obj.text, gt_text)
    except Exception as e:
        print(e)
        return OCRMetrics(Ratio(0, 0), Ratio(0, 0), Ratio(0, 0), Ratio(0, 0))

    if output_dir is not None:
        write_text_file(page_obj.text, output_dir / (name + ".hyp"))
        write_text_file(gt_text, output_dir / (name + ".ref"))
        write_json_file(page_metrics.dict, output_dir / (name + "__metrics.json"))
    
    return page_metrics


def evaluate_multiple_pages(
    pipeline_outputs: list, 
    gt_xml_paths: list, 
    output_dir: Path = None,
):
    metrics_list = []

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for pred, xml_path in zip(pipeline_outputs, gt_xml_paths):
        if output_dir is not None:
            img_metric_path = output_dir / (Path(xml_path).stem + "__metrics.json")
            if img_metric_path.exists():
                page_metrics = read_metric_dict(img_metric_path)
                metrics_list.append(page_metrics)
                continue
        
        page_metrics = evaluate_one_page(pred, xml_path)
        # print(f"Metrics: {page_metrics.float}")
        if page_metrics is not None:
            metrics_list.append(page_metrics)
        else:
            continue

    # Averaging metrics across all pages
    if metrics_list == []:
        print("No metrics found")
        return None
    
    avg_metrics: OCRMetrics = sum(metrics_list)
    if output_dir is not None:
        write_json_file(avg_metrics.dict, output_dir / "avg_metrics.json")

    return avg_metrics