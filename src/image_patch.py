import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import numpy as np

from src.vision_models.biomedclip_model import BioMedClipModel
from src.vision_models.medsam_model import MedSAMModel
from src.vision_models.cxr_hybridgnet_segmentation_model import CXRHybridGNetSegmentationModel
from src.agent.explainer import Explainer
from src.vision_models.braintumordetection_model import BrainTumorDetectionModel
from src.vision_models.uwmgi_segmentation_model import UWMGISegmentationModel
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
        self.braintumordetection_model = BrainTumorDetectionModel(output_dir=self.outputs_dir)
        self.uwmgi_segmentation_model = UWMGISegmentationModel(output_dir=self.outputs_dir)
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
    
    def verify_property(self, image_path: str, query: str):
        outputs = self.explainer(image_path, query)
        return outputs

    def segment_lungs_heart(self, image_path: str):
        result = self.cxr_segmentation_model(image_path)
        return result
    
    def detect_brain_tumor(self, image_path: str):
        # First, detect brain tumors using YOLO model
        result = {}
        detect = self.braintumordetection_model(image_path)

        # Save the detection-only path for reference
        result['detection_path'] = detect.get('path')
        result['detections'] = detect.get('detections', [])
        result['num_detections'] = detect.get('num_detections')
        result['segmentations'] = []

        # If MedSAM segmentation is requested and there are detections
        if detect['num_detections'] > 0:
            # Iterate through each detection and add segmentation mask
            for i, detection in enumerate(result['detections']):
                bbox = detection['bbox']  # [x1, y1, x2, y2]

                # Use MedSAM to segment the tumor based on bounding box
                try:
                    # MedSAM expects boxes in [x1, y1, x2, y2] format
                    seg_result = self.medsam_model(image_path, bbox)

                    # Also store in segmentations list
                    result['segmentations'].append({
                        'detection_index': i,
                        'bbox': bbox,
                        'mask_path': seg_result['mask_path'],
                        'overlay_path': seg_result['overlay_path']
                    })

                except Exception as e:
                    print(f"Error segmenting detection {i}: {e}")

        return result

    def segment_bowel_stomach(self, image_path: str):
        result = self.uwmgi_segmentation_model(image_path)
        return result
    
if __name__ == "__main__":
    # Example usage
    model = ImagePatch()
    sample_image_path = "/src/data/vqa_rad/images/img_143.jpg"
    result = model.detect_brain_tumor(sample_image_path)
    print(result)

    