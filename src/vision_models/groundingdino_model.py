import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from base_model import BaseModel  


class GroundingDINOModel(BaseModel):
    def __init__(self, model_path: str = "IDEA-Research/grounding-dino-tiny"):
        super().__init__()
        self.load_model(model_path)

    def load_model(self, model_path: str):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device)

    def forward(self, image_path: str, text_labels: str, threshold: float):
        img = Image.open(image_path)

        inputs = self.processor(images=img, text=text_labels, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            target_sizes=[img.size[::-1]]
        )

        result = results[0]
        return result

if __name__ == "__main__":
    model = GroundingDINOModel()
    result = model.forward("/Users/uncpham/Repo/Medical-Assistant/src/static/anh_meo_hai_huoc1.jpg", [["a cat"]], 0.4)
    img = Image.open("/Users/uncpham/Repo/Medical-Assistant/src/static/anh_meo_hai_huoc1.jpg")
    draw = ImageDraw.Draw(img)
    for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
        box_list = box.tolist()
        draw.rectangle(box_list, outline="red", width=3)
        box_rounded = [round(x, 2) for x in box_list]
        print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box_rounded}")
    img.save("/Users/uncpham/Repo/Medical-Assistant/src/static/output.jpg")
    print("Image with boxes saved as output.jpg")