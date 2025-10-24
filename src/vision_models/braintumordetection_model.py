import os
import sys
import cv2

# Add paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.join(os.path.dirname(__file__), "brain_tumor_detection_yolov8"))

from src.vision_models.brain_tumor_detection_yolov8.app import model
from src.vision_models.base_model import BaseModel
from src.constants.env import STATIC_FOLDER

class BrainTumorDetectionModel(BaseModel):
    def __init__(self, output_dir: str = None):
        super().__init__()  # Initialize BaseModel to set self.device
        self.model = model
        self.output_dir = output_dir if output_dir is not None else STATIC_FOLDER

    def __call__(self, image_path: str, return_annotated: bool = True):
        # YOLOv8 optimized inference parameters
        results = self.model.predict(
            image_path,
            conf=0.3,      # Confidence threshold for medical accuracy
            iou=0.5,       # IoU threshold for non-max suppression
            max_det=100,   # Maximum detections per image
            augment=False, # Disable augmentation for consistent results
            verbose=False
        )

        if not return_annotated:
            return results

        # Get first result
        result = results[0]

        # YOLOv8 enhanced plotting with medical-appropriate styling
        annotated_frame = result.plot(
            conf=True,        # Show confidence scores
            line_width=3,     # Thicker lines for better visibility
            font_size=14,     # Larger font for medical clarity
            labels=True,      # Show class labels
            boxes=True,       # Show bounding boxes
            probs=False       # Hide probability distributions
        )

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Get base filename from input image
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save annotated image
        overlay_filename = f'{base_filename}_brain_tumor_detection_overlay.png'
        overlay_path = os.path.join(self.output_dir, overlay_filename)

        # Convert RGB to BGR for cv2.imwrite
        cv2.imwrite(overlay_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        # Extract detection details
        detections = []
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            class_names = result.names if hasattr(result, 'names') else {}

            for conf, cls, bbox in zip(boxes.conf.cpu().numpy(),
                                       boxes.cls.cpu().numpy(),
                                       boxes.xyxy.cpu().numpy()):
                class_name = class_names.get(int(cls), f"Class {int(cls)}")
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': bbox.tolist()
                })

        return {
            'path': overlay_path,
            'detections': detections,
            'num_detections': len(detections),
            'class_names': result.names if hasattr(result, 'names') else {}
        }
    
if __name__ == "__main__":
    # Example usage
    btd_model = BrainTumorDetectionModel()
    result = btd_model("/Users/uncpham/Repo/Medical-Assistant/src/data/vqa_rad/images/img_121.jpg")
    print(f"Overlay path: {result['path']}")
    print(f"Number of detections: {result['num_detections']}")
    print(f"Detections: {result['detections']}")
    print(f"Class names: {result['class_names']}")



