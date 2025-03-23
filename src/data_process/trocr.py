import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from torch.utils.data import Dataset
from pagexml.parser import parse_pagexml_file
from src.data_process.running_text import RunningTextDatasetBuilder


class TrOCRLineDataset(Dataset):
    def __init__(self, img_xml_pairs: list[list] | list[tuple], use_cache: bool = True):
        super().__init__()
        self.builder = RunningTextDatasetBuilder()
        self._img_xml_pairs = img_xml_pairs
        self._idx_to_data = self._index(img_xml_pairs)
        self.use_cache = use_cache
        self.cache = {}

    def _index(self, img_xml_pairs):
        """Parse xml files and create a list containing (img, xml, line_idx)"""
        idx_to_data = []
        for img, xml in img_xml_pairs:
            try:
                xml_content = parse_pagexml_file(xml)
            except Exception as e:
                print(e)
                continue
            
            lines_data = xml_content.get_lines()
            
            # Image path, XML path, line index
            idx_to_data += [(img, xml, line_idx) for line_idx in range(len(lines_data))]  

        return idx_to_data

    def __len__(self):
        return len(self._idx_to_data)

    def _cache(self, img, xml):
        data = list(self.builder.process_one_page(img, xml))
        self.cache[img] = data

    def __getitem__(self, idx):
        # This can return None
        img, xml, line_idx = self._idx_to_data[idx]

        if self.use_cache:
            if img in self.cache:
                return self.cache[img][line_idx]
            else:
                self._cache(img, xml)
                return self.cache[img][line_idx]
        else:
            data = self.builder.process_one_line(img, xml, line_idx)
            return data
    
    

def create_trocr_collate_fn(processor, device):
    def func(batch):
        images = [data["image"] for data in batch]
        texts = [data["answer"] for data in batch]
        
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        labels = processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)

        return dict(
            pixel_values=pixel_values, 
            labels=labels,
        )

    return func