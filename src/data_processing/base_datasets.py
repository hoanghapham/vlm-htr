
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from abc import ABC, abstractmethod
from typing import Iterable

from src.data_processing.visual_tasks import IMAGE_EXTENSIONS
from src.file_tools import list_files
from src.data_processing.utils import XMLParser


class BaseImgXMLDataset(ABC):
    """Base dataset that load data from folders containing images & XML pairs
    The folder structure should be something like this:
    parent_folder/
        child_folder/
            images/
                image1.jpg
                image2.jpg
            page_xmls/
                image1.xml
                image2.xml
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.xmlparser = XMLParser()
        self._all_img_paths, self._all_xml_paths = self.validate_data_dir(data_dir)
        

    def validate_data_dir(self, data_dir: str | Path):
        img_paths = list_files(data_dir, IMAGE_EXTENSIONS)
        xml_paths = list_files(data_dir, [".xml"])
        
        # Validate that the img and xml files match
        matched = set([path.stem for path in img_paths]).intersection(set([path.stem for path in xml_paths]))
        assert len(img_paths) == len(xml_paths) == len(matched) > 0, \
            f"Length invalid, or mismatch img-xml pairs: {len(img_paths)} images, {len(xml_paths)} XML files, {len(matched)} matches"
        return img_paths, xml_paths

    @property
    @abstractmethod
    def nsamples(self):
        pass

    @abstractmethod
    def validate_and_load(self, img_paths, xml_paths):
        pass

    @abstractmethod
    def _get_one(self, idx):
        pass

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            return self._get_one(idx)
        else:
            return [self._get_one(i) for i in range(idx.start or 0, idx.stop or -1, idx.step or 1) if i < self.nsamples]

    def select(self, indices: Iterable):
        for idx in indices:
            yield self.__getitem__(idx)
