"""
Test script to visualize bounding box merging for overlapping boxes.
Shows before and after merging boxes with IoU > 0.5
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import colorsys


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    Boxes are in COCO format: [x, y, width, height]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to corner format
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def merge_boxes(box1: Dict, box2: Dict) -> Dict:
    """
    Merge two boxes by taking the union of their bounding boxes.
    Returns merged box in same format as input.
    """
    x1, y1, w1, h1 = box1['bbox']
    x2, y2, w2, h2 = box2['bbox']

    # Calculate corners
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)

    # Create merged box
    merged = {
        'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
        'category_id': box1['category_id'],  # Keep first box's category
        'area': (x_max - x_min) * (y_max - y_min),
        'merged_from': [box1.get('id', 'merged'), box2.get('id', 'merged')]
    }

    return merged


def merge_overlapping_boxes(annotations: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Merge boxes that have IoU > threshold.
    Uses greedy approach: merge boxes with highest IoU first.
    """
    boxes = [ann.copy() for ann in annotations]
    merged = []

    while boxes:
        # Take first box
        current_box = boxes.pop(0)

        # Find all boxes that overlap with current box
        to_merge = [current_box]
        remaining = []

        for box in boxes:
            # Only merge boxes of same category
            if box['category_id'] == current_box['category_id']:
                iou = calculate_iou(current_box['bbox'], box['bbox'])
                if iou > iou_threshold:
                    to_merge.append(box)
                else:
                    remaining.append(box)
            else:
                remaining.append(box)

        # Merge all overlapping boxes
        if len(to_merge) > 1:
            merged_box = to_merge[0]
            for box in to_merge[1:]:
                merged_box = merge_boxes(merged_box, box)
            merged.append(merged_box)
        else:
            merged.append(current_box)

        boxes = remaining

    return merged


def get_color_for_category(category_id: int, num_categories: int = 14) -> Tuple[int, int, int]:
    """Generate distinct color for each category using HSV color space."""
    hue = (category_id * 360 / num_categories) % 360
    rgb = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.9)
    return tuple(int(c * 255) for c in rgb)


def draw_boxes_on_image(image: np.ndarray, annotations: List[Dict],
                        categories: Dict[int, str], title: str = "") -> np.ndarray:
    """Draw bounding boxes on image with labels."""
    img_with_boxes = image.copy()

    for ann in annotations:
        x, y, w, h = [int(v) for v in ann['bbox']]
        cat_id = ann['category_id']
        cat_name = categories.get(cat_id, f"Class {cat_id}")

        # Get color for category
        color = get_color_for_category(cat_id)

        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 3)

        # Prepare label
        label = cat_name
        if 'merged_from' in ann:
            label += f" (merged: {len(ann['merged_from'])})"

        # Draw label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_label = max(y, label_size[1] + 10)
        cv2.rectangle(img_with_boxes,
                     (x, y_label - label_size[1] - 5),
                     (x + label_size[0], y_label + baseline),
                     color, -1)

        # Draw label text
        cv2.putText(img_with_boxes, label, (x, y_label - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_with_boxes


def visualize_bbox_merging(image_id: int = 13407, iou_threshold: float = 0.5):
    """
    Main function to visualize bounding box merging.

    Args:
        image_id: ID of image to visualize (default: 13407, which has 48 annotations)
        iou_threshold: IoU threshold for merging boxes
    """
    # Paths
    project_root = Path(__file__).parent
    ann_file = project_root / "src/data/vinbigdata-cxr-ad-coco/annotations/instances_train.json"
    img_dir = project_root / "src/data/vinbigdata-cxr-ad-coco/images/train"

    # Load annotations
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Get image info
    image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
    if not image_info:
        print(f"Image ID {image_id} not found!")
        return

    print(f"\nImage: {image_info['file_name']}")
    print(f"Size: {image_info['width']}x{image_info['height']}")

    # Get annotations for this image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    print(f"Number of original annotations: {len(annotations)}")

    # Count by category
    from collections import Counter
    cat_counts = Counter(ann['category_id'] for ann in annotations)
    print("\nAnnotations by category (before merging):")
    for cat_id, count in sorted(cat_counts.items()):
        print(f"  {categories[cat_id]}: {count}")

    # Load image
    img_path = img_dir / image_info['file_name']
    if not img_path.exists():
        print(f"Warning: Image file not found at {img_path}")
        print("Creating dummy image for demonstration...")
        image = np.ones((image_info['height'], image_info['width'], 3), dtype=np.uint8) * 200
    else:
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Merge overlapping boxes
    print(f"\nMerging boxes with IoU > {iou_threshold}...")
    merged_annotations = merge_overlapping_boxes(annotations, iou_threshold)
    print(f"Number of annotations after merging: {len(merged_annotations)}")
    print(f"Reduction: {len(annotations) - len(merged_annotations)} boxes merged")

    # Count by category after merging
    cat_counts_merged = Counter(ann['category_id'] for ann in merged_annotations)
    print("\nAnnotations by category (after merging):")
    for cat_id, count in sorted(cat_counts_merged.items()):
        reduction = cat_counts[cat_id] - count
        print(f"  {categories[cat_id]}: {count} (reduced by {reduction})")

    # Draw boxes
    print("\nGenerating visualizations...")
    img_before = draw_boxes_on_image(image, annotations, categories, "Before Merging")
    img_after = draw_boxes_on_image(image, merged_annotations, categories, "After Merging")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Before merging
    axes[0].imshow(img_before)
    axes[0].set_title(f'Before Merging\n{len(annotations)} bounding boxes',
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # After merging
    axes[1].imshow(img_after)
    axes[1].set_title(f'After Merging (IoU > {iou_threshold})\n{len(merged_annotations)} bounding boxes',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Add overall title
    fig.suptitle(f'Bounding Box Merging Visualization\nImage: {image_info["file_name"]}',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = project_root / f"bbox_merging_visualization_img{image_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Close plot instead of showing (for non-interactive environments)
    plt.close()

    # Print some statistics about merged boxes
    print("\n" + "="*60)
    print("MERGING STATISTICS")
    print("="*60)
    merged_count = sum(1 for ann in merged_annotations if 'merged_from' in ann)
    print(f"Total boxes that are result of merging: {merged_count}")

    if merged_count > 0:
        print("\nMerged box details:")
        for i, ann in enumerate(merged_annotations):
            if 'merged_from' in ann:
                cat_name = categories[ann['category_id']]
                num_merged = len(ann['merged_from'])
                print(f"  {i+1}. {cat_name}: merged from {num_merged} boxes")


if __name__ == "__main__":
    import sys

    # Default parameters
    image_id = 13407  # Image with 48 annotations
    iou_threshold = 0.5

    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        image_id = int(sys.argv[1])
    if len(sys.argv) > 2:
        iou_threshold = float(sys.argv[2])

    print("="*60)
    print("BOUNDING BOX MERGING VISUALIZATION")
    print("="*60)
    print(f"Image ID: {image_id}")
    print(f"IoU Threshold: {iou_threshold}")
    print("="*60)

    visualize_bbox_merging(image_id, iou_threshold)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nTo test with different images or IoU threshold:")
    print(f"  python {sys.argv[0]} <image_id> <iou_threshold>")
    print("\nExample:")
    print(f"  python {sys.argv[0]} 3414 0.3")