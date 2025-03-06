
from PIL import Image

image_path = "/content/drive/MyDrive/Projects/thesis-data/poliskammare-sample/small-sample/SCR-20250305-oefv.png"
image = Image.open(image_path).convert("RGB")

prompt = "<DocVQA>Print out the text in the image"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<VQA>", image_size=(image.width, image.height))

print(generated_text)
print(parsed_answer)
