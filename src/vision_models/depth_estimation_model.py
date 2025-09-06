import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import cv2
from PIL import Image
import numpy as np

from src.vision_models.base_model import BaseModel


class MiDaSModel(BaseModel):
    def __init__(self, model_name="Intel/dpt-large"):
        super().__init__()
        self.load_model(model_name)

    def load_model(self, model_name="Intel/dpt-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)

    def forward(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # resize về cùng kích thước ảnh gốc
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()
        return depth

if __name__ == "__main__":
    model = MiDaSModel()
    depth = model.forward(
        "/Users/uncpham/Repo/Medical-Assistant/src/static/anh_meo_hai_huoc1.jpg"
    )

    print("Depth map generated.")
    
    # Normalize and save as image
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255
    depth_image = Image.fromarray(depth_normalized.astype(np.uint8), mode='L')
    depth_image.save("/Users/uncpham/Repo/Medical-Assistant/src/static/depth_output.jpg")
    print("Depth image saved as depth_output.jpg")
