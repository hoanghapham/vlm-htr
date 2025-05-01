import sys
from pathlib import Path
from htrflow.evaluate import CER, WER, BagOfWords

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))


def compute_ocr_metrics(gt_text: str, pred_text: str, return_type: str = "ratio") -> dict:

    assert return_type in ["ratio", "str"], "return_type must be either 'ratio' or 'str'"

    cer = CER()
    wer = WER()
    bow = BagOfWords()

    cer_value = cer.compute(gt_text, pred_text)["cer"],
    wer_value = wer.compute(gt_text, pred_text)["wer"],
    bow_hits_value = bow.compute(gt_text, pred_text)["bow_hits"],
    bow_extras_value = bow.compute(gt_text, pred_text)["bow_extras"],

    result = dict()

    if return_type == "ratio":
        result = dict(
            cer=cer_value,
            wer=wer_value,
            bow_hits=bow_hits_value,
            bow_extras=bow_extras_value
        )
    elif return_type == "str":
        result = dict(
            cer = {"str": str(cer_value), "float": float(cer_value)},
            wer = {"str": str(wer_value), "float": float(wer_value)},
            bow_hits = {"str": str(bow_hits_value), "float": float(bow_hits_value)},
            bow_extras = {"str": str(bow_extras_value), "float": float(bow_extras_value)}
        )

    return result