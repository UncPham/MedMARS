import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from typing import List
import cv2

from src.vision_models.base_model import BaseModel
from src.constants.env import STATIC_FOLDER


class MedSAMModel(BaseModel):
    def __init__(self, output_dir: str = None):
        super().__init__()
        self.load_model()
        self.output_dir = output_dir if output_dir is not None else STATIC_FOLDER

    def __call__(self, image_path: str, boxes: List[float]):
        raw_image = Image.open(image_path).convert("RGB")
        # Format boxes as required: [[[x1, y1, x2, y2]]]
        formatted_boxes = [[boxes]]
        inputs = self.processor(raw_image, input_boxes=formatted_boxes, return_tensors="pt").to(
            self.device
        )

        outputs = self.model(**inputs, multimask_output=False)
        probs = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.sigmoid().cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
            binarize=False,
        )

        # Get the mask (threshold at 0.5)
        mask = (probs[0] > 0.5).squeeze().numpy().astype(np.uint8) * 255

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Get base filename from input image
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save mask
        mask_filename = f'{base_filename}_medsam_mask.png'
        mask_path = os.path.join(self.output_dir, mask_filename)
        cv2.imwrite(mask_path, mask)

        # Create overlay image
        img_array = np.array(raw_image)
        overlay = img_array.copy()

        # Apply yellow semi-transparent mask where mask is True
        mask_bool = mask > 0
        overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([255, 255, 0]) * 0.5

        # Save overlay
        overlay_filename = f'{base_filename}_medsam_overlay.png'
        overlay_path = os.path.join(self.output_dir, overlay_filename)
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))

        return {
            'mask_path': mask_path,
            'overlay_path': overlay_path
        }

    @staticmethod
    def show_mask(mask, ax, random_color: bool = False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def load_model(self):
        self.model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base").to(
            self.device
        )
        self.processor = SamProcessor.from_pretrained(
            "flaviagiammarino/medsam-vit-base"
        )


if __name__ == "__main__":
    model = MedSAMModel()
    image_path = (
        "/Users/uncpham/Repo/Medical-Assistant/src/data/vqa_rad/images/img_0.jpg"
    )
    input_boxes = [95.0, 255.0, 190.0, 350.0]

    result = model(image_path, input_boxes)
    print(f"Mask path: {result['mask_path']}")
    print(f"Overlay path: {result['overlay_path']}")
