
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