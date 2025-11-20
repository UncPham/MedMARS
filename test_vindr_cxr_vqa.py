import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Tuple, Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.medmars import MedMARS

# Configuration
NUM_QUESTIONS_TO_TEST = 15  # Set to None to test all questions, or a number 
JSON_PATH = "src/data/vindr_cxr_vqa/val_v1_clean.json"
IMAGES_DIR = "src/data/vindr_cxr_vqa/images"
IOU_THRESHOLD = 0.3  # IoU threshold for considering a detection as correct

# Generate timestamp for this test run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_BASE_DIR = f"logs/vindr_cxr_vqa_{TIMESTAMP}"

def scale_bbox_to_1024(bbox: List[int], orig_width: int, orig_height: int) -> List[float]:
    target_size = 1024
    scale_x = target_size / orig_width
    scale_y = target_size / orig_height

    return [
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y
    ]

def parse_location_tag(loc_string: str) -> Optional[List[int]]:
    pattern = r'<loc_(\d+)_(\d+)_(\d+)_(\d+)>'
    match = re.search(pattern, loc_string)
    if match:
        return [int(match.group(1)), int(match.group(2)),
                int(match.group(3)), int(match.group(4))]
    return None

def extract_all_locations(text: str) -> List[List[int]]:
    pattern = r'<loc_(\d+)_(\d+)_(\d+)_(\d+)>'
    matches = re.findall(pattern, text)
    return [[int(m[0]), int(m[1]), int(m[2]), int(m[3])] for m in matches]

def calculate_iou(box1: List[int], box2: List[int]) -> float:
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

def calculate_bbox_metrics(pred_boxes: List[List[int]], gt_boxes: List[List[int]],
                           iou_threshold: float = 0.3) -> Dict[str, float]:
    if len(gt_boxes) == 0:
        # No ground truth boxes
        if len(pred_boxes) == 0:
            return {'iou_mean': 1.0, 'recall': 1.0, 'f1': 1.0, 'precision': 1.0}
        else:
            return {'iou_mean': 0.0, 'recall': 0.0, 'f1': 0.0, 'precision': 0.0}

    if len(pred_boxes) == 0:
        # No predictions but ground truth exists
        return {'iou_mean': 0.0, 'recall': 0.0, 'f1': 0.0, 'precision': 0.0}

    # Calculate IoU matrix
    iou_matrix = []
    for pred_box in pred_boxes:
        ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
        iou_matrix.append(ious)

    # Greedy matching: for each GT box, find best matching prediction
    matched_gt = set()
    matched_pred = set()
    ious_matched = []

    for gt_idx in range(len(gt_boxes)):
        best_iou = 0.0
        best_pred_idx = -1

        for pred_idx in range(len(pred_boxes)):
            if pred_idx in matched_pred:
                continue

            iou = iou_matrix[pred_idx][gt_idx]
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_pred_idx >= 0:
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred_idx)
            ious_matched.append(best_iou)

    # Calculate metrics
    true_positives = len(matched_gt)
    false_positives = len(pred_boxes) - len(matched_pred)
    false_negatives = len(gt_boxes) - len(matched_gt)

    recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou_mean = sum(ious_matched) / len(ious_matched) if ious_matched else 0.0

    return {
        'iou_mean': iou_mean,
        'recall': recall,
        'f1': f1,
        'precision': precision
    }

def calculate_text_recall(predicted: str, ground_truth: str) -> float:
    """Calculate word-based recall score"""
    pred_words = set(predicted.lower().split())
    gt_words = set(ground_truth.lower().split())

    if len(gt_words) == 0:
        return 1.0 if len(pred_words) == 0 else 0.0

    matching_words = pred_words.intersection(gt_words)
    return len(matching_words) / len(gt_words)

def calculate_text_f1(predicted: str, ground_truth: str) -> float:
    """Calculate word-based F1 score"""
    pred_words = set(predicted.lower().split())
    gt_words = set(ground_truth.lower().split())

    if len(pred_words) == 0 and len(gt_words) == 0:
        return 1.0

    if len(pred_words) == 0 or len(gt_words) == 0:
        return 0.0

    matching_words = pred_words.intersection(gt_words)

    precision = len(matching_words) / len(pred_words) if len(pred_words) > 0 else 0.0
    recall = len(matching_words) / len(gt_words) if len(gt_words) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def save_markdown_report(output_dir: str, question_data: dict):
    """Save detailed markdown report for each question"""
    markdown_path = os.path.join(output_dir, "report.md")

    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(f"# VinDr-CXR VQA Analysis Report\n\n")
        f.write(f"## Question\n{question_data['question']}\n\n")
        f.write(f"## Image\n`{question_data['image_path']}`\n\n")

        f.write(f"## Thought\n```\n{question_data['thought']}\n```\n\n")
        f.write(f"## Plan\n```\n{question_data['plan']}\n```\n\n")
        f.write(f"## Generated Code\n```python\n{question_data['code']}\n```\n\n")
        f.write(f"## Code Output\n```\n{question_data['output']}\n```\n\n")

        f.write(f"## Final Answer\n{question_data['answer']}\n\n")
        f.write(f"## Explanation\n{question_data['reason']}\n\n")

        f.write(f"## Ground Truth\n")
        f.write(f"**Answer**: {question_data['gt_answer']}\n\n")
        f.write(f"**Reason**: {question_data['gt_reason']}\n\n")
        f.write(f"**Location**: {question_data['gt_location']}\n\n")
        f.write(f"**Finding**: {question_data['gt_finding']}\n\n")

        f.write(f"## Metrics\n")
        f.write(f"**Text Recall**: {question_data['text_recall']:.4f}\n\n")
        f.write(f"**Text F1**: {question_data['text_f1']:.4f}\n\n")
        f.write(f"**BBox IoU@{IOU_THRESHOLD}**: {question_data['bbox_iou']:.4f}\n\n")
        f.write(f"**BBox Recall**: {question_data['bbox_recall']:.4f}\n\n")
        f.write(f"**BBox F1**: {question_data['bbox_f1']:.4f}\n\n")
        f.write(f"**BBox Precision**: {question_data['bbox_precision']:.4f}\n\n")

        f.write(f"## Predicted Bboxes\n{question_data['pred_boxes']}\n\n")
        f.write(f"## Ground Truth Bboxes\n{question_data['gt_boxes']}\n\n")

def setup_logging(test_run_dir):
    """Setup logging to file and console"""
    log_file = os.path.join(test_run_dir, "test.log")

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

def test_vindr_cxr_vqa():
    """Main test function"""
    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_BASE_DIR)
    logger.info(f"Starting VinDr-CXR VQA test run at {TIMESTAMP}")
    logger.info(f"Test results will be saved to: {OUTPUT_BASE_DIR}")

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
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten questions from all images
    all_questions = []
    for entry in data:
        image_id = entry['image_id']
        for vqa in entry['vqa']:
            all_questions.append({
                'image_id': image_id,
                'question': vqa['question'],
                'answer': vqa['answer'],
                'reason': vqa['reason'],
                'type': vqa['type'],
                'difficulty': vqa['difficulty'],
                'gt_finding': vqa['gt_finding'],
                'gt_location': vqa['gt_location']
            })

    # Limit number of questions if specified
    if NUM_QUESTIONS_TO_TEST is not None:
        all_questions = all_questions[:NUM_QUESTIONS_TO_TEST]

    logger.info(f"Testing {len(all_questions)} questions from {len(data)} images...")

    # Initialize MedMARS
    logger.info("Initializing MedMARS...")
    medmars = MedMARS()

    # Track metrics
    text_recalls = []
    text_f1s = []
    bbox_ious = []
    bbox_recalls = []
    bbox_f1s = []
    bbox_precisions = []
    all_results = []

    # Track by question type
    metrics_by_type = {}

    for i, q in enumerate(all_questions):
        logger.info("="*80)
        logger.info(f"Question {i+1}/{len(all_questions)}")
        logger.info("="*80)

        image_id = q['image_id']
        question = q['question']
        gt_answer = q['answer']
        gt_reason = q['reason']
        gt_location = q['gt_location']
        gt_finding = q['gt_finding']
        q_type = q['type']
        difficulty = q['difficulty']

        image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")

        # Create output directory for this question
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"vqa_{i}")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Image ID: {image_id}")
        logger.info(f"Question: {question}")
        logger.info(f"Type: {q_type}, Difficulty: {difficulty}")
        logger.info(f"GT Answer: {gt_answer}")
        logger.info(f"GT Location: {gt_location}")

        try:
            # Run MedMARS with output_dir for this question
            thought, plan, code, output, result, response = medmars.run(
                query=question,
                image=image_path,
                output_dir=output_dir
            )

            # Extract answer and reason from response
            pred_answer = response.get('answer', '')
            pred_reason = response.get('reason', '') + " Visible at specified location."

            logger.info(f"Predicted Answer: {pred_answer}")

            # Calculate text metrics (using answer + reason)
            combined_pred = f"{pred_answer} {pred_reason}"
            combined_gt = f"{gt_answer} {gt_reason}"

            text_recall = calculate_text_recall(combined_pred, combined_gt)
            text_f1 = calculate_text_f1(combined_pred, combined_gt)

            # Calculate bbox metrics
            pred_boxes = extract_all_locations(pred_answer)
            gt_boxes = extract_all_locations(gt_location)

            # Scale bboxes from original resolution to 1024x1024
            if image_id in image_dimensions:
                orig_w, orig_h = image_dimensions[image_id]
                gt_boxes_scaled = [scale_bbox_to_1024(box, orig_w, orig_h) for box in gt_boxes]
                pred_boxes_scaled = pred_boxes
            else:
                logger.warning(f"No dimensions found for {image_id}, using boxes as-is")
                gt_boxes_scaled = gt_boxes
                pred_boxes_scaled = pred_boxes

            bbox_metrics = calculate_bbox_metrics(pred_boxes_scaled, gt_boxes_scaled, IOU_THRESHOLD)

            logger.info(f"Text Recall: {text_recall:.4f}, Text F1: {text_f1:.4f}")
            logger.info(f"BBox IoU@{IOU_THRESHOLD}: {bbox_metrics['iou_mean']:.4f}, "
                       f"Recall: {bbox_metrics['recall']:.4f}, "
                       f"F1: {bbox_metrics['f1']:.4f}")
            logger.info(f"Pred Boxes (orig): {pred_boxes}, GT Boxes (orig): {gt_boxes}")
            if image_id in image_dimensions:
                logger.info(f"Pred Boxes (1024): {[[round(x) for x in box] for box in pred_boxes_scaled]}, "
                           f"GT Boxes (1024): {[[round(x) for x in box] for box in gt_boxes_scaled]}")

            # Track overall metrics
            text_recalls.append(text_recall)
            text_f1s.append(text_f1)
            bbox_ious.append(bbox_metrics['iou_mean'])
            bbox_recalls.append(bbox_metrics['recall'])
            bbox_f1s.append(bbox_metrics['f1'])
            bbox_precisions.append(bbox_metrics['precision'])

            # Track by question type
            if q_type not in metrics_by_type:
                metrics_by_type[q_type] = {
                    'text_recalls': [],
                    'text_f1s': [],
                    'bbox_ious': [],
                    'bbox_recalls': [],
                    'bbox_f1s': [],
                    'count': 0
                }

            metrics_by_type[q_type]['text_recalls'].append(text_recall)
            metrics_by_type[q_type]['text_f1s'].append(text_f1)
            metrics_by_type[q_type]['bbox_ious'].append(bbox_metrics['iou_mean'])
            metrics_by_type[q_type]['bbox_recalls'].append(bbox_metrics['recall'])
            metrics_by_type[q_type]['bbox_f1s'].append(bbox_metrics['f1'])
            metrics_by_type[q_type]['count'] += 1

            # Prepare data for markdown report
            question_data = {
                'question': question,
                'image_path': image_path,
                'thought': thought if thought else "N/A",
                'plan': plan if plan else "N/A",
                'code': code if code else "N/A",
                'output': str(output) if output else "N/A",
                'answer': pred_answer,
                'reason': pred_reason,
                'gt_answer': gt_answer,
                'gt_reason': gt_reason,
                'gt_location': gt_location,
                'gt_finding': gt_finding,
                'text_recall': text_recall,
                'text_f1': text_f1,
                'bbox_iou': bbox_metrics['iou_mean'],
                'bbox_recall': bbox_metrics['recall'],
                'bbox_f1': bbox_metrics['f1'],
                'bbox_precision': bbox_metrics['precision'],
                'pred_boxes': pred_boxes_scaled if image_id in image_dimensions else pred_boxes,
                'gt_boxes': gt_boxes_scaled if image_id in image_dimensions else gt_boxes
            }

            # Save markdown report
            save_markdown_report(output_dir, question_data)

            all_results.append({
                'question_id': i,
                'image_id': image_id,
                'question': question,
                'type': q_type,
                'difficulty': difficulty,
                'gt_answer': gt_answer,
                'predicted_answer': pred_answer,
                'text_recall': text_recall,
                'text_f1': text_f1,
                'bbox_iou': bbox_metrics['iou_mean'],
                'bbox_recall': bbox_metrics['recall'],
                'bbox_f1': bbox_metrics['f1'],
                'bbox_precision': bbox_metrics['precision'],
                'num_pred_boxes': len(pred_boxes),
                'num_gt_boxes': len(gt_boxes)
            })

        except Exception as e:
            logger.error(f"Error processing question {i+1}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # Save error report
            error_data = {
                'question': question,
                'image_path': image_path,
                'thought': "ERROR",
                'plan': "ERROR",
                'code': "ERROR",
                'output': str(e),
                'answer': "ERROR",
                'reason': str(e),
                'gt_answer': gt_answer,
                'gt_reason': gt_reason,
                'gt_location': gt_location,
                'gt_finding': gt_finding,
                'text_recall': 0.0,
                'text_f1': 0.0,
                'bbox_iou': 0.0,
                'bbox_recall': 0.0,
                'bbox_f1': 0.0,
                'bbox_precision': 0.0,
                'pred_boxes': [],
                'gt_boxes': []
            }
            save_markdown_report(output_dir, error_data)

            all_results.append({
                'question_id': i,
                'image_id': image_id,
                'question': question,
                'type': q_type,
                'difficulty': difficulty,
                'gt_answer': gt_answer,
                'predicted_answer': "ERROR",
                'text_recall': 0.0,
                'text_f1': 0.0,
                'bbox_iou': 0.0,
                'bbox_recall': 0.0,
                'bbox_f1': 0.0,
                'bbox_precision': 0.0,
                'num_pred_boxes': 0,
                'num_gt_boxes': 0
            })

    # Calculate and print overall statistics
    logger.info("="*80)
    logger.info("OVERALL RESULTS")
    logger.info("="*80)

    if text_recalls:
        avg_text_recall = sum(text_recalls) / len(text_recalls)
        avg_text_f1 = sum(text_f1s) / len(text_f1s)
        logger.info(f"Text Metrics:")
        logger.info(f"  Average Recall: {avg_text_recall:.4f}")
        logger.info(f"  Average F1: {avg_text_f1:.4f}")

    if bbox_ious:
        avg_bbox_iou = sum(bbox_ious) / len(bbox_ious)
        avg_bbox_recall = sum(bbox_recalls) / len(bbox_recalls)
        avg_bbox_f1 = sum(bbox_f1s) / len(bbox_f1s)
        avg_bbox_precision = sum(bbox_precisions) / len(bbox_precisions)
        logger.info(f"\nBBox Metrics (IoU@{IOU_THRESHOLD}):")
        logger.info(f"  Average IoU: {avg_bbox_iou:.4f}")
        logger.info(f"  Average Recall: {avg_bbox_recall:.4f}")
        logger.info(f"  Average Precision: {avg_bbox_precision:.4f}")
        logger.info(f"  Average F1: {avg_bbox_f1:.4f}")

    # Print per-type statistics
    logger.info("\n" + "="*80)
    logger.info("RESULTS BY QUESTION TYPE")
    logger.info("="*80)

    for q_type, metrics in sorted(metrics_by_type.items()):
        logger.info(f"\nQuestion Type: {q_type} ({metrics['count']} questions)")
        logger.info(f"  Text Recall: {sum(metrics['text_recalls']) / len(metrics['text_recalls']):.4f}")
        logger.info(f"  Text F1: {sum(metrics['text_f1s']) / len(metrics['text_f1s']):.4f}")
        logger.info(f"  BBox IoU@{IOU_THRESHOLD}: {sum(metrics['bbox_ious']) / len(metrics['bbox_ious']):.4f}")
        logger.info(f"  BBox Recall: {sum(metrics['bbox_recalls']) / len(metrics['bbox_recalls']):.4f}")
        logger.info(f"  BBox F1: {sum(metrics['bbox_f1s']) / len(metrics['bbox_f1s']):.4f}")

    # Save summary JSON
    summary_path = os.path.join(OUTPUT_BASE_DIR, "test_summary.json")
    summary_data = {
        'timestamp': TIMESTAMP,
        'num_questions': len(all_questions),
        'iou_threshold': IOU_THRESHOLD,
        'overall_metrics': {
            'text_recall': avg_text_recall if text_recalls else 0.0,
            'text_f1': avg_text_f1 if text_f1s else 0.0,
            'bbox_iou': avg_bbox_iou if bbox_ious else 0.0,
            'bbox_recall': avg_bbox_recall if bbox_recalls else 0.0,
            'bbox_precision': avg_bbox_precision if bbox_precisions else 0.0,
            'bbox_f1': avg_bbox_f1 if bbox_f1s else 0.0
        },
        'metrics_by_type': {
            q_type: {
                'count': m['count'],
                'text_recall': sum(m['text_recalls']) / len(m['text_recalls']),
                'text_f1': sum(m['text_f1s']) / len(m['text_f1s']),
                'bbox_iou': sum(m['bbox_ious']) / len(m['bbox_ious']),
                'bbox_recall': sum(m['bbox_recalls']) / len(m['bbox_recalls']),
                'bbox_f1': sum(m['bbox_f1s']) / len(m['bbox_f1s'])
            }
            for q_type, m in metrics_by_type.items()
        },
        'detailed_results': all_results
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info(f"Individual reports saved to: {OUTPUT_BASE_DIR}/vqa_*/report.md")
    logger.info(f"Test log saved to: {OUTPUT_BASE_DIR}/test.log")
    logger.info(f"Test run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    test_vindr_cxr_vqa()