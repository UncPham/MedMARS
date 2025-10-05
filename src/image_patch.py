import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from src.vision_models.groundingdino_model import GroundingDINOModel
from src.vision_models.biomedclip_model import BioMedClipModel
from src.vision_models.midas_model import MiDaSModel
from src.vision_models.medclipsam_model import MedCLIPSAMModel
from src.vision_models.medsam_model import MedSAMModel


class ImagePatch:
    def __init__(self):
        # self.groundingdino_model = GroundingDINOModel()
        self.biomedclip_model = BioMedClipModel()
        self.midas_model = MiDaSModel()
        self.medclipsam_model = MedCLIPSAMModel()
        self.medsam_model = MedSAMModel()

    def get_bounding_boxes(self, image_path: str, text_labels: str, threshold: float):
        result = self.groundingdino_model(image_path, text_labels, threshold)
        boxes = result["boxes"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        return boxes, labels, scores

    def get_depth_map(self, image_path: str):
        depth_map = self.midas_model.forward(image_path)
        return depth_map

    def get_segmentation_masks(self, image_path: str, boxes, labels):
        masks = self.medclipsam_model(image_path, boxes, labels)
        return masks

    def get_medical_segmentation(self, image_path: str):
        medical_masks = self.medsam_model(image_path)
        return medical_masks