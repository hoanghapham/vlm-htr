import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

    
def create_collate_fn(processor, device):
    def func(batch):
        # Filter None item in the batch. In the worst case, all items are None
        batch = list(filter(lambda x: x is not None, batch))

        images = [data["image"] for data in batch]
        texts = [data["transcription"] for data in batch]
        
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        labels = processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)

        return dict(
            pixel_values=pixel_values, 
            labels=labels,
        )

    return func