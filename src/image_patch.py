import os
import sys

# Get absolute path of this file's directory
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

# Add project root to path
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add parent directories so DEIM and HybridGNet can be imported as packages
vision_models_dir = os.path.join(current_dir, "vision_models")
hybridgnet_path = os.path.join(vision_models_dir, "Chest_x_ray_HybridGNet_Segmentation")

# Insert at the beginning to ensure priority
# Add vision_models so DEIM can be imported as "DEIM.engine.core"
if vision_models_dir not in sys.path:
    sys.path.insert(0, vision_models_dir)
# Add HybridGNet for its internal imports
if hybridgnet_path not in sys.path:
    sys.path.insert(0, hybridgnet_path)

from src.vision_models.biomedclip_model import BioMedClipModel
from src.vision_models.medsam_model import MedSAMModel
from src.vision_models.cxr_hybridgnet_segmentation_model import CXRHybridGNetSegmentationModel
from src.vision_models.deim_model import DEIMModel
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
        self.deim_model = DEIMModel(output_dir=self.outputs_dir)
        self.explainer = Explainer()
    
    def best_image_match(self, images_path: list[str], labels: list[str]) -> dict:
        results = self.biomedclip_model(images_path, labels)

        return results

    def classification_chest(self, image_path: str):
        outputs = self.biomedclip_model([image_path], CHESTMNIST_LABEL)
        label_scores = outputs[image_path.split('/')[-1]]

        # Get all labels with confidence > 0.4
        detected_labels = [label for label, score in label_scores.items() if score > 0.4]

        # Return list of detected labels, or None if nothing detected
        return detected_labels if detected_labels else None
    
    def verify_property(self, list_image_path: list[str], query: str):
        outputs = self.explainer(query, list_image_path)
        return outputs

    def segment_lungs_heart(self, image_path: str):
        result = self.cxr_segmentation_model(image_path)
        return result
    
    def detect_chest_abnormality(self, image_path: str):
        deim_results = self.deim_model(image_path)

        # If no detections, return early
        if len(deim_results['boxes']) == 0:
            return {
                'detection': deim_results,
                'segmentation': []
            }

        # Segment each detected abnormality
        segmentations = []
        for box, label_name, score in zip(
            deim_results['boxes'],
            deim_results['label_names'],
            deim_results['scores']
        ):
            # Convert box to list format [x1, y1, x2, y2]
            box_list = box.tolist()
            print(f"Segmenting {label_name} with box {box_list}...")

            # Run MedSAM segmentation
            label = label_name.replace(" ", "_").lower()
            seg_result = self.medsam_model(image_path, box_list, label=label)

            # Add metadata
            seg_result['abnormality'] = label_name
            seg_result['box'] = box_list

            segmentations.append(seg_result)

        return {
            'detection': deim_results,
            'segmentations': segmentations
        }

if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("ImagePatch Demo - DEIM Detection + MedSAM Segmentation")
    print("=" * 80)

    model = ImagePatch()
    sample_image_path = "/Users/uncpham/Repo/Medical-Assistant/src/data/vindr_cxr_vqa/images/0a1aef5326b7b24378c6692f7a454e52.jpg"

    # Option 1: Just detect abnormalities
    # results = model.detect_chest_abnormality(sample_image_path)
    results = model.best_image_match(sample_image_path, "Cardiomegaly")

    print(f"results: {results}")

    