import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from typing import List

from src.vision_models.base_model import BaseModel


class MedSAMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.load_model()

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
        return probs

    @staticmethod
    def show_mask(mask, ax, random_color):
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
    raw_image = Image.open(image_path).convert("RGB")
    input_boxes = [95.0, 255.0, 190.0, 350.0]

    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2
            )
        )

    probs = model(image_path, input_boxes)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.array(raw_image))
    show_box(input_boxes, ax[0])
    ax[0].set_title("Input Image and Bounding Box")
    ax[0].axis("off")
    ax[1].imshow(np.array(raw_image))
    
    model.show_mask(mask=probs[0] > 0.5, ax=ax[1], random_color=False)
    show_box(input_boxes, ax[1])
    ax[1].set_title("MedSAM Segmentation")
    ax[1].axis("off")
    plt.show()
