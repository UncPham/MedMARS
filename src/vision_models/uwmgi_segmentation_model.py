import os
import sys
import cv2
import numpy as np
from PIL import Image

# Add paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.join(os.path.dirname(__file__), "UWMGI_Medical_Image_Segmentation"))

import torchvision.transforms as TF

from src.vision_models.UWMGI_Medical_Image_Segmentation.app import get_model, predict, Configs
from src.vision_models.base_model import BaseModel
from src.constants.env import STATIC_FOLDER


class UWMGISegmentationModel(BaseModel):
    """Medical Image Segmentation Model for UW-Madison GI Tract Dataset."""

    CLASS_COLORS = {
        "Large bowel": (0, 0, 255),    # Red
        "Small bowel": (23, 154, 0),   # Green
        "Stomach": (255, 127, 0)       # Blue
    }

    def __init__(self, output_dir: str = None):
        """Initialize the UWMGI Segmentation Model."""
        super().__init__()
        self.output_dir = output_dir if output_dir is not None else STATIC_FOLDER

        # Load model using existing function
        model_path = os.path.join(os.path.dirname(__file__),
                                  "UWMGI_Medical_Image_Segmentation",
                                  "segformer_trained_weights")
        self.model = get_model(model_path=model_path, num_classes=Configs.NUM_CLASSES)
        self.model.to(self.device)
        self.model.eval()

        # Setup preprocessing
        self.preprocess = TF.Compose([
            TF.Resize(size=Configs.IMAGE_SIZE[::-1]),
            TF.ToTensor(),
            TF.Normalize(Configs.MEAN, Configs.STD, inplace=True),
        ])

    def __call__(self, image_path: str, return_annotated: bool = True, alpha: float = 0.5):
        """Perform segmentation on the input image."""
        # Load image
        input_image = Image.open(image_path).convert("RGB")

        # Use existing predict function
        _, seg_info = predict(input_image, model=self.model,
                             preprocess_fn=self.preprocess, device=self.device)

        if not return_annotated:
            return {'masks': {class_name: mask for mask, class_name in seg_info}}

        os.makedirs(self.output_dir, exist_ok=True)

        # Create annotated overlay image
        original_image = cv2.imread(image_path)
        overlay = np.zeros_like(original_image, dtype=np.uint8)
        segmentation_info = []
        mask_images = {}

        for mask, class_name in seg_info:
            # Create individual mask visualization
            mask_vis = self._create_mask_image(original_image, mask, class_name)
            mask_images[class_name] = mask_vis

            if mask.any():
                color = self.CLASS_COLORS[class_name]
                overlay[mask] = color
                percentage = (mask.sum() / mask.size) * 100
                segmentation_info.append({
                    'class': class_name,
                    'percentage': float(percentage),
                    'color': color
                })

        # Blend overlay
        annotated_frame = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)

        # Save all images and collect paths
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save overlay image
        overlay_path = os.path.join(self.output_dir, f'{base_filename}_bowel_stomach_overlay.png')
        cv2.imwrite(overlay_path, annotated_frame)

        # Save individual mask images
        paths = {'overlay_path': overlay_path}
        for class_name, mask_img in mask_images.items():
            safe_name = class_name.lower().replace(' ', '_')
            mask_path = os.path.join(self.output_dir, f'{base_filename}_{safe_name}.png')
            cv2.imwrite(mask_path, mask_img)
            paths[f'{safe_name}_mask_path'] = mask_path

        # Return only paths and segmentation info
        result_dict = {
            'overlay_path': overlay_path,
            'large_bowel_mask_path': paths['large_bowel_mask_path'],
            'small_bowel_mask_path': paths['small_bowel_mask_path'],
            'stomach_mask_path': paths['stomach_mask_path'],
            'segmentation_info': segmentation_info
        }

        return result_dict

    def _create_mask_image(self, original_image, mask, class_name):
        """Create a visualization image for a single mask."""
        # Create black background
        mask_vis = np.zeros_like(original_image)

        # Apply color where mask is True
        color = self.CLASS_COLORS[class_name]
        mask_vis[mask] = color

        return mask_vis



if __name__ == "__main__":
    # Example usage
    model = UWMGISegmentationModel()

    # Test with a sample image
    test_image = "/Users/uncpham/Repo/Medical-Assistant/src/data/vqa_rad/images/img_25.jpg"

    
    result = model(test_image, return_annotated=True, alpha=0.5)

    print(result)