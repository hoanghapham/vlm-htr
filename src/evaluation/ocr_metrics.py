import sys
from pathlib import Path
from typing import Self

from htrflow.evaluate import CER, WER, BagOfWords

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from src.data_types import Ratio


class OCRMetrics():
    def __init__(self, cer: Ratio, wer: Ratio, bow_hits: Ratio, bow_extras: Ratio):
        self.cer = cer
        self.wer = wer
        self.bow_hits = bow_hits
        self.bow_extras = bow_extras

    def __repr__(self):
        return str(self.dict)

    def __str__(self):
        return str(self.dict)
    
    def __add__(self, other: Self):
        return OCRMetrics(
            self.cer + other.cer, 
            self.wer + other.wer, 
            self.bow_hits + other.bow_hits, 
            self.bow_extras + other.bow_extras
        )
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    @property
    def dict(self):
        result = dict(
            cer = {"str": str(self.cer), "float": float(self.cer)},
            wer = {"str": str(self.wer), "float": float(self.wer)},
            bow_hits = {"str": str(self.bow_hits), "float": float(self.bow_hits)},
            bow_extras = {"str": str(self.bow_extras), "float": float(self.bow_extras)}
        )
        return result

    @property
    def float_str(self):
        result = dict(
            cer = f"{float(self.cer):.4f}",
            wer = f"{float(self.wer):.4f}",
            bow_hits = f"{float(self.bow_hits):.4f}",
            bow_extras = f"{float(self.bow_extras):.4f}"
        )
        return result
    
    @property
    def fraction_str(self):
        result = dict(
            cer = str(self.cer),
            wer = str(self.wer),
            bow_hits = str(self.bow_hits),
            bow_extras = str(self.bow_extras)
        )
        return result
    

def compute_ocr_metrics(pred_text: str, gt_text: str) -> OCRMetrics:

    cer = CER()
    wer = WER()
    bow = BagOfWords()

    cer = cer.compute(gt_text, pred_text)["cer"]
    wer = wer.compute(gt_text, pred_text)["wer"]
    bow_hits = bow.compute(gt_text, pred_text)["bow_hits"]
    bow_extras = bow.compute(gt_text, pred_text)["bow_extras"]

    result = OCRMetrics(cer, wer, bow_hits, bow_extras)
    return result
