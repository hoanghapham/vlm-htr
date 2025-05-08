
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from htrflow.evaluate import Ratio
from src.file_tools import read_json_file
from src.data_processing.utils import XMLParser
from src.evaluation.ocr_metrics import compute_ocr_metrics, OCRMetrics
from src.file_tools import write_text_file, write_json_file


def read_metric_dict(metric_dict_path: str | Path, ) -> OCRMetrics:
    metric_dict  = read_json_file(metric_dict_path)
    cer         = Ratio(*metric_dict["cer"]["str"].split("/"))
    wer         = Ratio(*metric_dict["wer"]["str"].split("/"))
    bow_hits    = Ratio(*metric_dict["bow_hits"]["str"].split("/"))
    bow_extras  = Ratio(*metric_dict["bow_extras"]["str"].split("/"))
    return OCRMetrics(cer, wer, bow_hits, bow_extras)


def evaluate_pipeline(
    pipeline_outputs: list, 
    gt_xml_paths: list, 
    output_dir: Path = None,
):
    xml_parser = XMLParser()
    metrics_list = []

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)

    for pred, xml_path in zip(pipeline_outputs, gt_xml_paths):
        # Write predicted text in .hyp extension to be used with E2EHTREval
        if output_dir is not None:
            img_metric_path = output_dir / (Path(xml_path).stem + "__metrics.json")
            if img_metric_path.exists():
                page_metrics = read_metric_dict(img_metric_path)
                metrics_list.append(page_metrics)
                continue

        if output_dir is not None:
            write_text_file(pred.text, output_dir / (Path(xml_path).stem + ".hyp"))

        # Get lines from xml
        # Write ground truth in .ref extension to be used with E2EHTREval
        gt_lines    = xml_parser.get_lines(xml_path)
        gt_text     = " ".join([line["transcription"] for line in gt_lines])

        if output_dir is not None:
            write_text_file(gt_text, output_dir / (Path(xml_path).stem + ".ref"))

        # Evaluation
        try:
            page_metrics = compute_ocr_metrics(pred.text, gt_text)
            metrics_list.append(page_metrics)
        except Exception as e:
            print(e)
            continue
        
        if output_dir is not None:
            write_json_file(page_metrics.dict, output_dir / (Path(xml_path).stem + "__metrics.json"))
        # print(f"Page metrics: {page_metrics.result_float}")

    # Averaging metrics across all pages
    if metrics_list == []:
        print("No metrics found")
        return None
    
    avg_metrics: OCRMetrics = sum(metrics_list)
    print(f"Avg. metrics: {avg_metrics.float}")
    if output_dir is not None:
        write_json_file(avg_metrics.dict, output_dir / "avg_metrics.json")

    return metrics_list