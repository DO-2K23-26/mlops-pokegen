from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-small-printed",
    use_fast=False
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-small-printed"
)

image = Image.open("image.png").convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print(text)