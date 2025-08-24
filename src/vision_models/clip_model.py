import torch
from transformers import CLIPModel, CLIPProcessor
from typing import List
from PIL import Image

from vision_models.base_model import BaseModel


class CLIPModel(BaseModel):
    def __init__(self, model_path: str):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_path)

    def forward(self, image_path: str, labels: List[str]):
        raw_image = Image.open(image_path).convert("RGB")
        # Sử dụng CLIP để tìm patch phù hợp nhất
        inputs = self.clip_processor(
            text=[labels], images=raw_image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs
