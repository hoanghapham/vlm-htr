import typing
from pathlib import Path
from torch.utils.data import Dataset
from datasets import concatenate_datasets, load_from_disk


class HTRDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<SwedishHTR>Print out the text in this image"
        answer = example["transcription"]
        image = example['image'].convert("RGB")
        return dict(
            question=question, 
            answer=answer, 
            image=image
        )
    
    def select(self, indices: typing.Iterable):
        subset = [self.data[int(idx)] for idx in indices]
        return HTRDataset(subset)


def create_dset_from_paths(path_list: list[str | Path]):
    dsets = []
    for path in path_list:
        dsets.append(load_from_disk(path))
    data = concatenate_datasets(dsets)
    return HTRDataset(data)


def create_collate_fn(processor, device):
    def func(batch):
        questions = [data["question"] for data in batch]
        answers = [data["answer"] for data in batch]
        images = [data["image"] for data in batch]
        
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
        return dict(
            input_ids=inputs["input_ids"], 
            pixel_values=inputs["pixel_values"], 
            labels=labels,
        )

    return func


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