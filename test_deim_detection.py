import os
import sys
import json
import re
from datetime import datetime
import logging
from typing import List, Dict
import numpy as np

# Add paths for DEIM and HybridGNet modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add vision_models directories
vision_models_dir = os.path.join(current_dir, "src", "vision_models")
deim_path = os.path.join(vision_models_dir, "DEIM")
hybridgnet_path = os.path.join(vision_models_dir, "Chest_x_ray_HybridGNet_Segmentation")

if vision_models_dir not in sys.path:
    sys.path.insert(0, vision_models_dir)
if deim_path not in sys.path:
    sys.path.insert(0, deim_path)
if hybridgnet_path not in sys.path:
    sys.path.insert(0, hybridgnet_path)

from src.vision_models.deim_model import DEIMModel

# Configuration
NUM_IMAGES_TO_TEST = None  # Set to None to test all images, or a number like 100
JSON_PATH = "src/data/vindr_cxr_vqa/val_v1_clean.json"
IMAGES_DIR = "src/data/vindr_cxr_vqa/images"
CONF_THRESHOLD = 0.3  # Confidence threshold for detections
IOU_THRESHOLD = 0.3   # IoU threshold for matching predictions with ground truth

# Generate timestamp for this test run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"logs/deim_evaluation_{TIMESTAMP}"

def parse_location_tag(loc_string: str) -> List[int]:
    """
    Parse location tag from format <loc_x1_y1_x2_y2> to [x1, y1, x2, y2]

    Args:
        loc_string: String containing location tag like "<loc_691_1375_1653_1831>"

    Returns:
        List of [x1, y1, x2, y2] coordinates or None if not found
    """
    pattern = r'<loc_(\d+)_(\d+)_(\d+)_(\d+)>'
    match = re.search(pattern, loc_string)
    if match:
        return [int(match.group(1)), int(match.group(2)),
                int(match.group(3)), int(match.group(4))]
    return None

def scale_bbox_to_1024(bbox: List[int], orig_width: int, orig_height: int) -> List[float]:
    """
    Scale bounding box from original image size to 1024x1024

    Images were resized by stretching (ignoring aspect ratio) from original size to 1024x1024.
    GT boxes are in original resolution, so we need to scale them to match DEIM's output.

    Args:
        bbox: [x1, y1, x2, y2] in original resolution
        orig_width: Original image width
        orig_height: Original image height

    Returns:
        [x1, y1, x2, y2] in 1024x1024 resolution
    """
    target_size = 1024
    scale_x = target_size / orig_width
    scale_y = target_size / orig_height

    return [
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y
    ]

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate IoU (Intersection over Union) between two bounding boxes

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score (0.0 to 1.0)
    """
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate intersection area
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def match_predictions_to_gt(pred_boxes: List[List[int]],
                            pred_labels: List[str],
                            pred_scores: List[float],
                            gt_boxes: List[List[int]],
                            gt_labels: List[str],
                            iou_threshold: float = 0.3) -> Dict:
    """
    Match predicted boxes to ground truth boxes using IoU threshold

    Returns:
        Dict with matched predictions, false positives, and false negatives
    """
    matches = []  # List of (pred_idx, gt_idx, iou, label)
    matched_gt = set()
    matched_pred = set()

    # For each GT box, find best matching prediction with same label
    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        best_iou = 0.0
        best_pred_idx = -1

        for pred_idx, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            # Skip if already matched or labels don't match
            if pred_idx in matched_pred or pred_label != gt_label:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_pred_idx >= 0:
            matches.append({
                'pred_idx': best_pred_idx,
                'gt_idx': gt_idx,
                'iou': best_iou,
                'label': gt_label,
                'pred_score': pred_scores[best_pred_idx]
            })
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred_idx)

    # False positives: predictions that didn't match any GT
    false_positives = []
    for pred_idx in range(len(pred_boxes)):
        if pred_idx not in matched_pred:
            false_positives.append({
                'pred_idx': pred_idx,
                'box': pred_boxes[pred_idx],
                'label': pred_labels[pred_idx],
                'score': pred_scores[pred_idx]
            })

    # False negatives: GT boxes that didn't match any prediction
    false_negatives = []
    for gt_idx in range(len(gt_boxes)):
        if gt_idx not in matched_gt:
            false_negatives.append({
                'gt_idx': gt_idx,
                'box': gt_boxes[gt_idx],
                'label': gt_labels[gt_idx]
            })

    return {
        'matches': matches,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'num_gt': len(gt_boxes),
        'num_pred': len(pred_boxes)
    }

def calculate_average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) using 11-point interpolation
    """
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.
    return ap

def calculate_metrics_per_class(all_predictions: List[Dict],
                                 all_ground_truths: List[Dict],
                                 class_name: str,
                                 iou_threshold: float = 0.3) -> Dict:
    """
    Calculate detection rate (recall-only) for a specific class.

    This metric focuses ONLY on whether GT findings are detected correctly.
    False positives (extra predictions) do NOT affect the score.

    This is appropriate when:
    - GT annotations may be incomplete (e.g., VQA data)
    - We only care about detecting what's labeled, not penalizing extra detections
    """
    # Collect all predictions and GTs for this class
    class_predictions = []
    class_gts = []

    for img_preds in all_predictions:
        for i, (box, label, score) in enumerate(zip(
            img_preds['boxes'],
            img_preds['labels'],
            img_preds['scores']
        )):
            if label == class_name:
                class_predictions.append({
                    'box': box,
                    'score': score,
                    'image_id': img_preds['image_id']
                })

    for img_gts in all_ground_truths:
        for box, label in zip(img_gts['boxes'], img_gts['labels']):
            if label == class_name:
                class_gts.append({
                    'box': box,
                    'image_id': img_gts['image_id']
                })

    num_gt = len(class_gts)

    if num_gt == 0:
        return {
            'detection_rate': 0.0,
            'recall': 0.0,
            'num_gt': 0,
            'num_pred': len(class_predictions),
            'tp': 0,
            'fn': 0
        }

    if len(class_predictions) == 0:
        return {
            'detection_rate': 0.0,
            'recall': 0.0,
            'num_gt': num_gt,
            'num_pred': 0,
            'tp': 0,
            'fn': num_gt
        }

    # For each GT, find if there's ANY matching prediction
    # We DON'T care about false positives
    matched_gts = set()

    for gt_idx, gt in enumerate(class_gts):
        best_iou = 0.0
        best_pred_idx = -1

        # Find best matching prediction for this GT
        for pred_idx, pred in enumerate(class_predictions):
            if gt['image_id'] != pred['image_id']:
                continue

            iou = calculate_iou(pred['box'], gt['box'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_pred_idx >= 0:
            matched_gts.add(gt_idx)

    # Calculate detection rate
    tp = len(matched_gts)
    fn = num_gt - tp
    detection_rate = tp / num_gt

    return {
        'detection_rate': detection_rate,
        'recall': detection_rate,  # Same as detection rate
        'num_gt': num_gt,
        'num_pred': len(class_predictions),
        'tp': tp,
        'fn': fn
    }

def setup_logging(output_dir):
    """Setup logging to file and console"""
    log_file = os.path.join(output_dir, "evaluation.log")

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def test_deim_detection():
    """Main test function"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info(f"Starting DEIM detection evaluation at {TIMESTAMP}")
    logger.info(f"Results will be saved to: {OUTPUT_DIR}")
    logger.info(f"Confidence threshold: {CONF_THRESHOLD}")
    logger.info(f"IoU threshold: {IOU_THRESHOLD}")

    # Load COCO annotations to get original image dimensions
    logger.info("Loading COCO annotations for image dimensions...")
    coco_train_path = "src/data/vinbigdata-cxr-ad-coco/annotations/instances_train.json"
    coco_val_path = "src/data/vinbigdata-cxr-ad-coco/annotations/instances_val.json"

    image_dimensions = {}  # image_id -> (width, height)

    for coco_path in [coco_train_path, coco_val_path]:
        if os.path.exists(coco_path):
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
            for img in coco_data['images']:
                image_id = img['file_name'].replace('.jpg', '')
                image_dimensions[image_id] = (img['width'], img['height'])

    logger.info(f"Loaded dimensions for {len(image_dimensions)} images")

    # Read JSON file
    logger.info(f"Loading data from {JSON_PATH}")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract unique image_ids and their ground truth boxes
    image_data = {}  # image_id -> list of (finding, location)

    for entry in data:
        image_id = entry['image_id']
        if image_id not in image_data:
            image_data[image_id] = []

        for vqa in entry['vqa']:
            gt_finding = vqa.get('gt_finding')
            gt_location = vqa.get('gt_location')

            if gt_finding and gt_location:
                box = parse_location_tag(gt_location)
                if box:
                    # Scale box from original resolution to 1024x1024
                    if image_id in image_dimensions:
                        orig_w, orig_h = image_dimensions[image_id]
                        box = scale_bbox_to_1024(box, orig_w, orig_h)
                    else:
                        logger.warning(f"No dimensions found for {image_id}, using box as-is")

                    image_data[image_id].append({
                        'finding': gt_finding,
                        'box': box
                    })

    # Remove duplicates per image
    for image_id in image_data:
        # Use set to track unique (finding, box) pairs
        # Round coordinates to avoid floating point precision issues
        seen = set()
        unique_items = []
        for item in image_data[image_id]:
            # Round box coordinates to 2 decimal places for comparison
            rounded_box = tuple(round(coord, 2) for coord in item['box'])
            key = (item['finding'], rounded_box)
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        image_data[image_id] = unique_items

    logger.info(f"Found {len(image_data)} unique images with annotations")

    # Limit number of images if specified
    image_ids = list(image_data.keys())
    if NUM_IMAGES_TO_TEST is not None:
        image_ids = image_ids[:NUM_IMAGES_TO_TEST]

    logger.info(f"Testing on {len(image_ids)} images...")

    # Initialize DEIM model
    logger.info("Loading DEIM model...")
    deim_model = DEIMModel(output_dir=OUTPUT_DIR)

    # Track results
    all_predictions = []
    all_ground_truths = []
    image_results = []

    for idx, image_id in enumerate(image_ids):
        logger.info("="*80)
        logger.info(f"Image {idx+1}/{len(image_ids)}: {image_id}")
        logger.info("="*80)

        image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")

        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue

        # Get ground truth
        gt_items = image_data[image_id]
        gt_boxes = [item['box'] for item in gt_items]
        gt_labels = [item['finding'] for item in gt_items]

        logger.info(f"Ground truth: {len(gt_boxes)} boxes")
        for label, box in zip(gt_labels, gt_boxes):
            logger.info(f"  - {label}: {box}")

        # Run DEIM detection
        try:
            detection_result = deim_model(image_path)

            # Extract predictions above confidence threshold
            pred_boxes = []
            pred_labels = []
            pred_scores = []

            for box, label, score in zip(
                detection_result['boxes'],
                detection_result['label_names'],
                detection_result['scores']
            ):
                if score >= CONF_THRESHOLD:
                    pred_boxes.append(box.tolist())
                    pred_labels.append(label)
                    pred_scores.append(float(score))

            logger.info(f"Predictions: {len(pred_boxes)} boxes (conf >= {CONF_THRESHOLD})")
            for label, box, score in zip(pred_labels, pred_boxes, pred_scores):
                logger.info(f"  - {label}: {box} (conf: {score:.3f})")

            # Match predictions to ground truth
            match_result = match_predictions_to_gt(
                pred_boxes, pred_labels, pred_scores,
                gt_boxes, gt_labels,
                IOU_THRESHOLD
            )

            logger.info(f"Matches: {len(match_result['matches'])}")
            logger.info(f"False Positives: {len(match_result['false_positives'])}")
            logger.info(f"False Negatives: {len(match_result['false_negatives'])}")

            # Store results
            all_predictions.append({
                'image_id': image_id,
                'boxes': pred_boxes,
                'labels': pred_labels,
                'scores': pred_scores
            })

            all_ground_truths.append({
                'image_id': image_id,
                'boxes': gt_boxes,
                'labels': gt_labels
            })

            image_results.append({
                'image_id': image_id,
                'image_path': image_path,
                'num_gt': len(gt_boxes),
                'num_pred': len(pred_boxes),
                'num_matches': len(match_result['matches']),
                'num_fp': len(match_result['false_positives']),
                'num_fn': len(match_result['false_negatives']),
                'matches': match_result['matches'],
                'false_positives': match_result['false_positives'],
                'false_negatives': match_result['false_negatives']
            })

        except Exception as e:
            logger.error(f"Error processing image {image_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    # Calculate per-class metrics
    logger.info("\n" + "="*80)
    logger.info("CALCULATING PER-CLASS METRICS")
    logger.info("="*80)

    per_class_metrics = {}
    all_classes = set()

    for img_gt in all_ground_truths:
        all_classes.update(img_gt['labels'])

    for img_pred in all_predictions:
        all_classes.update(img_pred['labels'])

    logger.info(f"Found {len(all_classes)} classes: {sorted(all_classes)}")

    for class_name in sorted(all_classes):
        logger.info(f"\nCalculating metrics for class: {class_name}")
        metrics = calculate_metrics_per_class(
            all_predictions,
            all_ground_truths,
            class_name,
            IOU_THRESHOLD
        )
        per_class_metrics[class_name] = metrics
        logger.info(f"  Detection Rate: {metrics['detection_rate']:.4f} ({metrics['tp']}/{metrics['num_gt']})")
        logger.info(f"  GT boxes: {metrics['num_gt']}, Predictions: {metrics['num_pred']}")
        logger.info(f"  True Positives: {metrics['tp']}, False Negatives: {metrics['fn']}")

    # Calculate overall metrics
    logger.info("\n" + "="*80)
    logger.info("OVERALL METRICS (Recall-Only Mode)")
    logger.info("="*80)
    logger.info("NOTE: This evaluation ignores false positives.")
    logger.info("Only measures if GT findings are detected (appropriate for incomplete GT).")

    mean_detection_rate = np.mean([m['detection_rate'] for m in per_class_metrics.values()])
    total_tp = sum([m['tp'] for m in per_class_metrics.values()])
    total_gt = sum([m['num_gt'] for m in per_class_metrics.values()])
    total_fn = sum([m['fn'] for m in per_class_metrics.values()])

    logger.info(f"Mean Detection Rate: {mean_detection_rate:.4f}")
    logger.info(f"Total: {total_tp}/{total_gt} GT findings detected")
    logger.info(f"Total True Positives: {total_tp}")
    logger.info(f"Total False Negatives: {total_fn}")

    # Save results
    results_file = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    results_data = {
        'config': {
            'conf_threshold': CONF_THRESHOLD,
            'iou_threshold': IOU_THRESHOLD,
            'num_images': len(image_ids),
            'num_classes': len(all_classes),
            'evaluation_mode': 'recall_only',
            'note': 'False positives are ignored. Only GT detection rate is measured.'
        },
        'per_class_metrics': per_class_metrics,
        'overall_metrics': {
            'mean_detection_rate': mean_detection_rate,
            'total_gt': total_gt,
            'total_tp': total_tp,
            'total_fn': total_fn,
            'overall_detection_rate': total_tp / total_gt if total_gt > 0 else 0.0
        },
        'detailed_results': image_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Evaluation log saved to: {OUTPUT_DIR}/evaluation.log")
    logger.info(f"Evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    test_deim_detection()
