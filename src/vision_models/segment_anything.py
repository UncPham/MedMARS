import torch
from PIL import Image
from transformers import SamModel, SamProcessor
from typing import List, Tuple

from vision_models.base_model import BaseModel


class SegmentAnything(BaseModel):
    def __init__(self, model_path: str):
        super().__init__()
        self.load_model(model_path)

    def forward(
        self, image_path: str, input_boxes: List[Tuple[float, float, float, float]]
    ):
        """
        This method uses the SAM model to generate segmentation masks for objects specified by
        bounding box coordinates in the input image.

        Input:
        - image_path (str): Path to the image file to be segmented.
        - input_boxes (List[Tuple[float, float, float, float]]): List of bounding boxes in format (x1, y1, x2, y2)
          where coordinates are normalized between 0 and 1.

        Output:
        - outputs (dict): Dictionary containing SAM model outputs with keys:
            - pred_masks: Predicted segmentation masks as tensors
            - iou_predictions: IoU predictions for each mask
            - low_res_masks: Low-resolution masks for refinement
        """
        raw_image = Image.open(image_path).convert("RGB")
        # Convert input_boxes to the format expected by SAM processor: [[[x1, y1, x2, y2]]]
        formatted_boxes = [[list(box)] for box in input_boxes]
        inputs = self.sam_processor(
            raw_image, input_boxes=formatted_boxes, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        return outputs

    def load_model(self, model_path: str):
        self.sam_model = SamModel.from_pretrained(model_path).to(self.device)
        self.sam_processor = SamProcessor.from_pretrained(model_path)
