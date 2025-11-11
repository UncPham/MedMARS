import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import numpy as np

from src.vision_models.biomedclip_model import BioMedClipModel
from src.vision_models.medsam_model import MedSAMModel
from src.vision_models.cxr_hybridgnet_segmentation_model import CXRHybridGNetSegmentationModel
from src.agent.explainer import Explainer
from src.constants.env import STATIC_FOLDER
from src.constants.constants import CHESTMNIST_LABEL

class ImagePatch:
    def __init__(self, outputs_dir: str = None):
        # Set outputs directory - use STATIC_FOLDER if not specified
        self.outputs_dir = outputs_dir if outputs_dir is not None else str(STATIC_FOLDER)

        # Ensure output directory exists
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Initialize models
        self.biomedclip_model = BioMedClipModel()
        self.medsam_model = MedSAMModel(output_dir=self.outputs_dir)
        self.cxr_segmentation_model = CXRHybridGNetSegmentationModel(output_dir=self.outputs_dir)
        self.explainer = Explainer()
    
    def best_image_match(self, images_path: list[str], labels: list[str]) -> dict:
        results = self.biomedclip_model(images_path, labels)

        return results

    def classification_chest(self, image_path: str):
        outputs = self.biomedclip_model([image_path], CHESTMNIST_LABEL)
        label_scores = outputs[image_path.split('/')[-1]]
        best_label = max(label_scores.items(), key=lambda x: x[1])
        if best_label[1] > 0.4:
            return best_label[0]
        return None
    
    def verify_property(self, list_image_path: list[str], query: str):
        outputs = self.explainer(query, list_image_path)
        return outputs

    def segment_lungs_heart(self, image_path: str):
        result = self.cxr_segmentation_model(image_path)
        return result
    
if __name__ == "__main__":
    # Example usage
    model = ImagePatch()
    sample_image_path = "/src/data/vqa_rad/images/img_143.jpg"
    result = model.detect_brain_tumor(sample_image_path)
    print(result)

    