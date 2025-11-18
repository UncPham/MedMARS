"""
DEIM Model Wrapper for Inference
Provides a clean interface for using DEIM models
"""

import os
import sys

# Add vision_models directory to path so DEIM can be imported as a package
vision_models_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

if vision_models_dir not in sys.path:
    sys.path.insert(0, vision_models_dir)
if project_root not in sys.path:
    sys.path.insert(1, project_root)

# Set environment variables for single-GPU/CPU mode (no distributed training)
os.environ.setdefault('RANK', '0')
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '12355')

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, List
import colorsys

# Import DEIM modules
from DEIM.engine.core import YAMLConfig
from src.vision_models.base_model import BaseModel
from src.constants.env import DEIM_CHECKPOINT, DEIM_CONFIG
from src.constants.env import STATIC_FOLDER


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Compute intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def merge_overlapping_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, scores, labels

    # Group by class label
    unique_labels = np.unique(labels)

    merged_boxes_list = []
    merged_scores_list = []
    merged_labels_list = []

    for label in unique_labels:
        # Get boxes for this class
        class_mask = labels == label
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        # Track which boxes have been merged
        merged = np.zeros(len(class_boxes), dtype=bool)

        for i in range(len(class_boxes)):
            if merged[i]:
                continue

            # Find all boxes that overlap with current box
            overlapping_indices = [i]

            for j in range(i + 1, len(class_boxes)):
                if merged[j]:
                    continue

                iou = compute_iou(class_boxes[i], class_boxes[j])
                if iou > iou_threshold:
                    overlapping_indices.append(j)
                    merged[j] = True

            # Merge overlapping boxes using weighted average
            overlapping_boxes = class_boxes[overlapping_indices]
            overlapping_scores = class_scores[overlapping_indices]

            # Normalize scores to use as weights
            weights = overlapping_scores / overlapping_scores.sum()

            # Weighted average of box coordinates
            merged_box = np.average(overlapping_boxes, axis=0, weights=weights)

            # Use max confidence score
            merged_score = overlapping_scores.max()

            merged_boxes_list.append(merged_box)
            merged_scores_list.append(merged_score)
            merged_labels_list.append(label)

    if len(merged_boxes_list) == 0:
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(merged_boxes_list),
        np.array(merged_scores_list),
        np.array(merged_labels_list)
    )


class DEIMModel(BaseModel):
    # VinBigData chest X-ray class names
    CLASS_NAMES = [
        "Aortic enlargement",      # 0
        "Pleural thickening",      # 1
        "Pleural effusion",        # 2
        "Cardiomegaly",            # 3
        "Lung Opacity",            # 4
        "Nodule/Mass",             # 5
        "Consolidation",           # 6
        "Pulmonary fibrosis",      # 7
        "Infiltration",            # 8
        "Atelectasis",             # 9
        "Other lesion",            # 10
        "ILD",                     # 11
        "Pneumothorax",            # 12
        "Calcification"            # 13
    ]

    def __init__(
        self,
        config_path: str = DEIM_CONFIG,
        checkpoint_path: str = DEIM_CHECKPOINT,
        output_dir: str = STATIC_FOLDER,
        device: Optional[str] = None,
        input_size: int = 640
    ):
        super().__init__()

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.input_size = input_size
        self.output_dir = output_dir

        # Set device
        if device is not None:
            self.device = device

        # Model and config will be loaded lazily
        self.model = None
        self.cfg = None

        # Load model on initialization
        self.load_model()

    def load_model(self):
        # Initialize single-process distributed mode if not already initialized
        if not torch.distributed.is_initialized():
            try:
                torch.distributed.init_process_group(
                    backend='gloo',  # Use 'gloo' for CPU, 'nccl' for GPU
                    init_method='tcp://localhost:12355',
                    world_size=1,
                    rank=0
                )
                print("Initialized single-process distributed mode")
            except Exception as e:
                print(f"Warning: Could not initialize distributed mode: {e}")

        print(f"Loading config from {self.config_path}")
        # Override pretrained to False for inference (we're loading full checkpoint)
        self.cfg = YAMLConfig(
            self.config_path,
            resume=self.checkpoint_path,
            HGNetv2={'pretrained': False}  # Disable pretrained backbone loading
        )

        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        # Build model
        self.model = self.cfg.model.to(self.device)

        # Load weights (try both 'model_ema' and 'model')
        if 'model_ema' in checkpoint:
            print("Loading EMA model weights")
            self.model.load_state_dict(checkpoint['model_ema'])
        elif 'model' in checkpoint:
            print("Loading regular model weights")
            self.model.load_state_dict(checkpoint['model'])
        else:
            raise ValueError("No model weights found in checkpoint")

        self.model.eval()
        print(f"Model loaded successfully on {self.device}!")

        return self.model

    def preprocess(
        self,
        image: Union[str, Path]
    ) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Get original size
        orig_w, orig_h = img.size

        # Resize to input size
        img_resized = img.resize((self.input_size, self.input_size), Image.BILINEAR)

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1)
        img_tensor = img_tensor.float() / 255.0

        return img_tensor, (orig_w, orig_h), np.array(img)

    @torch.no_grad()
    def inference(self, image_tensor: torch.Tensor) -> Dict:
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Run model
        outputs = self.model(image_tensor)

        return outputs

    def postprocess(
        self,
        outputs: Dict,
        orig_size: Tuple[int, int],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        # Extract first item if outputs is a list (batch size = 1)
        output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        # Extract boxes, scores, labels - DEIM/D-FINE format
        try:
            if isinstance(output, dict) and 'pred_boxes' in output and 'pred_logits' in output:
                # DEIM/D-FINE format
                pred_boxes = output['pred_boxes']  # [batch, num_queries, 4]
                pred_logits = output['pred_logits']  # [batch, num_queries, num_classes]

                # Remove batch dimension (batch_size=1)
                pred_boxes = pred_boxes[0] if pred_boxes.dim() == 3 else pred_boxes
                pred_logits = pred_logits[0] if pred_logits.dim() == 3 else pred_logits

                # Convert logits to probabilities using sigmoid (for multi-label)
                pred_scores = torch.sigmoid(pred_logits)  # [num_queries, num_classes]

                # Get max score and corresponding label for each query
                scores, labels = pred_scores.max(dim=-1)  # [num_queries]

                boxes = pred_boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()

            elif isinstance(output, dict) and 'boxes' in output:
                # Standard format
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
            else:
                # Tensor format
                boxes = output[:, :4].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                labels = output[:, 5].cpu().numpy()

        except Exception as e:
            raise RuntimeError(f"Error extracting outputs: {e}")

        # Filter by confidence threshold
        keep = scores > conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Convert from cxcywh normalized to xyxy pixel coordinates
        orig_w, orig_h = orig_size
        boxes_xyxy = []

        for box in boxes:
            cx, cy, w, h = box

            # Denormalize to input size coordinates
            cx *= self.input_size
            cy *= self.input_size
            w *= self.input_size
            h *= self.input_size

            # Convert to xyxy format
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Scale to original image size
            x1 = x1 * orig_w / self.input_size
            y1 = y1 * orig_h / self.input_size
            x2 = x2 * orig_w / self.input_size
            y2 = y2 * orig_h / self.input_size

            boxes_xyxy.append([x1, y1, x2, y2])

        boxes_xyxy = np.array(boxes_xyxy)

        # Merge overlapping boxes with same class
        boxes_merged, scores_merged, labels_merged = merge_overlapping_boxes(
            boxes_xyxy, scores, labels, iou_threshold=iou_threshold
        )

        # Convert label IDs to label names
        label_names = [self.CLASS_NAMES[int(label)] for label in labels_merged]

        return {
            'boxes': boxes_merged,
            'scores': scores_merged,
            'labels': labels_merged,
            'label_names': label_names
        }

    def __call__(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5
    ) -> Dict:
        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # Preprocess
        img_tensor, orig_size, _ = self.preprocess(image)

        # Inference
        outputs = self.inference(img_tensor)

        # Postprocess
        results = self.postprocess(outputs, orig_size, conf_threshold, iou_threshold)

        # save overlays
        layer_results = self.create_and_save_layer_overlays(
            image=image,
            results=results,
            output_dir=self.output_dir
        )
        # Add overlay paths to results (only filenames, not full paths)
        results['overlay_paths'] = {class_name: os.path.basename(data['path']) for class_name, data in layer_results.items()}

        return results

    @staticmethod
    def generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            saturation = 0.8 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.9 + (i % 2) * 0.1  # Vary brightness slightly
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    def create_overlay(
        self,
        image: Union[str, Path, np.ndarray],
        results: Dict,
        show_labels: bool = True,
        show_scores: bool = True,
        line_width: int = 3,
        font_size: int = 20,
        specific_class: Optional[Union[int, str]] = None
    ) -> np.ndarray:
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype('uint8'))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Create drawing context
        overlay = img.copy()
        draw = ImageDraw.Draw(overlay, 'RGBA')

        # Generate colors for all classes
        colors = self.generate_colors(len(self.CLASS_NAMES))

        # Try to load a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Filter results if specific class is requested
        if specific_class is not None:
            if isinstance(specific_class, str):
                # Find class index by name
                try:
                    class_idx = self.CLASS_NAMES.index(specific_class)
                except ValueError:
                    raise ValueError(f"Class '{specific_class}' not found in CLASS_NAMES")
            else:
                class_idx = specific_class

            # Filter detections
            mask = results['labels'] == class_idx
            boxes = results['boxes'][mask]
            scores = results['scores'][mask]
            labels = results['labels'][mask]
            label_names = [results['label_names'][i] for i, m in enumerate(mask) if m]
        else:
            boxes = results['boxes']
            scores = results['scores']
            labels = results['labels']
            label_names = results['label_names']

        # Draw each detection
        for box, score, label, label_name in zip(boxes, scores, labels, label_names):
            x1, y1, x2, y2 = box
            color = colors[int(label)]

            # Draw only box outline (no fill)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Prepare label text
            text_parts = []
            if show_labels:
                text_parts.append(label_name)
            if show_scores:
                text_parts.append(f"{score:.2f}")

            if text_parts:
                text = " - ".join(text_parts)

                # Get text bounding box
                bbox = draw.textbbox((x1, y1), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Draw background for text
                text_bg = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]
                draw.rectangle(text_bg, fill=color + (200,))

                # Draw text
                draw.text((x1 + 2, y1 - text_height - 2), text, fill=(255, 255, 255), font=font)

        return np.array(overlay)

    def create_layer_overlays(
        self,
        image: Union[str, Path, np.ndarray],
        results: Dict,
        show_labels: bool = True,
        show_scores: bool = True,
        line_width: int = 3,
        font_size: int = 20
    ) -> Dict[str, np.ndarray]:
        layer_overlays = {}

        # Get unique classes in detections
        unique_labels = np.unique(results['labels'])

        for label in unique_labels:
            class_name = self.CLASS_NAMES[int(label)]
            overlay = self.create_overlay(
                image=image,
                results=results,
                show_labels=show_labels,
                show_scores=show_scores,
                line_width=line_width,
                font_size=font_size,
                specific_class=int(label)
            )
            layer_overlays[class_name] = overlay

        return layer_overlays

    def save_overlay(
        self,
        overlay: np.ndarray,
        output_path: Union[str, Path]
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.fromarray(overlay.astype('uint8'))
        img.save(output_path)

    def save_layer_overlays(
        self,
        layer_overlays: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
    ) -> Dict[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        for class_name, overlay in layer_overlays.items():
            # Create safe filename
            safe_name = class_name.replace('/', '_').replace(' ', '_')
            output_path = output_dir / f"overlay_bbox_{safe_name}.png"
            self.save_overlay(overlay, output_path)
            saved_paths[class_name] = str(output_path)

        return saved_paths

    def create_and_save_layer_overlays(
        self,
        image: Union[str, Path, np.ndarray],
        results: Dict,
        output_dir: Union[str, Path],
        show_labels: bool = True,
        show_scores: bool = True,
        line_width: int = 3,
        font_size: int = 20
    ) -> Dict[str, Dict[str, Union[np.ndarray, str]]]:
        # Create overlays
        layer_overlays = self.create_layer_overlays(
            image=image,
            results=results,
            show_labels=show_labels,
            show_scores=show_scores,
            line_width=line_width,
            font_size=font_size
        )

        # Save overlays and get paths
        saved_paths = self.save_layer_overlays(layer_overlays, output_dir)

        # Combine into single result
        result = {}
        for class_name in layer_overlays.keys():
            result[class_name] = {
                'image': layer_overlays[class_name],
                'path': saved_paths[class_name]
            }

        return result


if __name__ == '__main__':
    # Example usage
    print("DEIM Model Example Usage")
    print("=" * 60)

    model = DEIMModel(
        input_size=640
    )

    # Run inference on an image
    image_path = '/Users/uncpham/Repo/Medical-Assistant/src/data/vindr_cxr_vqa/images/0a1aef5326b7b24378c6692f7a454e52.jpg'

    # Inference with automatic overlay saving
    results = model(
        image_path,
        conf_threshold=0.3,
        iou_threshold=0.5,
    )

    print(f"results: {results}")

    print("\nInference Results:")
    print(f"Detected {len(results['boxes'])} objects")

    if results['overlay_paths']:
        print("\nOverlay paths saved:")
        for class_name, path in results['overlay_paths'].items():
            print(f"  - {class_name}: {path}")

    print("\nDetections:")
    for i, (box, score, label_name) in enumerate(zip(results['boxes'], results['scores'], results['label_names'])):
        print(f"  {i+1}. {label_name} (confidence: {score:.3f})")
        print(f"     Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
