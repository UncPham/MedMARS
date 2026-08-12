"""
DEIM Detector - detect abnormalities on chest X-rays and save an overlay

Usage:
    python src/vision_models/deim_detector.py <image_path> [output_path]
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vision_models.deim_model import DEIMModel


class DEIMDetector:
    """Simple detector for chest X-ray abnormalities"""

    def __init__(self):
        """Initialize the DEIM model"""
        self.model = DEIMModel()
        self.class_names = self.model.CLASS_NAMES

    @staticmethod
    def generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate a distinct color for each abnormality class"""
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            saturation = 0.85
            value = 0.95
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    def detect(self, image_path: str, conf_threshold: float = 0.3) -> Dict:
        """
        Detect abnormalities in a chest X-ray

        Args:
            image_path: Path to the image
            conf_threshold: Confidence threshold (default: 0.3)

        Returns:
            Detection results (boxes, scores, labels, label_names)
        """
        results = self.model(image_path, conf_threshold=conf_threshold)
        return results

    def create_combined_overlay(
        self,
        image_path: str,
        results: Dict,
        line_width: int = 3,
        font_size: int = 20
    ) -> np.ndarray:
        """
        Build an overlay showing ALL detected abnormalities on a single image

        Args:
            image_path: Path to the original image
            results: Output of detect()
            line_width: Bounding box thickness
            font_size: Label font size

        Returns:
            Overlay image as a numpy array
        """
        # Load the image
        img = Image.open(image_path).convert('RGB')
        overlay = img.copy()
        draw = ImageDraw.Draw(overlay, 'RGBA')

        # Build the per-class color palette
        colors = self.generate_colors(len(self.class_names))

        # Load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Draw every detection
        for box, score, label, label_name in zip(
            results['boxes'],
            results['scores'],
            results['labels'],
            results['label_names']
        ):
            x1, y1, x2, y2 = box
            color = colors[int(label)]

            # Draw the bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Draw the label and confidence score
            text = f"{label_name} - {score:.2f}"
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Text background
            text_bg = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]
            draw.rectangle(text_bg, fill=color + (220,))

            # Draw the text
            draw.text((x1 + 2, y1 - text_height - 2), text, fill=(0, 0, 0), font=font)

        return np.array(overlay)

    def detect_and_save(
        self,
        image_path: str,
        output_path: str,
        conf_threshold: float = 0.3
    ) -> Dict:
        """
        Full pipeline: detect abnormalities and save the overlay

        Args:
            image_path: Input image path
            output_path: Where to save the overlay
            conf_threshold: Confidence threshold

        Returns:
            Dictionary containing the results and related info
        """
        # Detect abnormalities
        print(f"Analyzing image: {image_path}")
        results = self.detect(image_path, conf_threshold)

        num_detections = len(results['boxes'])
        detected_diseases = list(set(results['label_names']))

        print(f"\nDetected {num_detections} abnormal regions:")
        for disease in detected_diseases:
            count = results['label_names'].count(disease)
            print(f"  - {disease}: {count} region(s)")

        # Build the overlay
        print("\nCreating overlay...")
        overlay = self.create_combined_overlay(image_path, results)

        # Save the overlay
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay.astype('uint8')).save(output_path)
        print(f"Overlay saved to: {output_path}")

        return {
            'detections': results,
            'overlay_path': str(output_path),
            'num_detections': num_detections,
            'detected_diseases': detected_diseases
        }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <image_path> [output_path]")
        sys.exit(1)

    IMAGE_PATH = sys.argv[1]
    OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else None
    CONF_THRESHOLD = 0.3

    # Derive an output path when none is given
    if OUTPUT_PATH is None:
        input_path = Path(IMAGE_PATH)
        output_dir = input_path.parent / "deim_results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_overlay.png"
    else:
        output_path = OUTPUT_PATH

    # Initialize the detector
    print("Initializing DEIM model...")
    detector = DEIMDetector()

    # Detect and save
    result = detector.detect_and_save(
        image_path=IMAGE_PATH,
        output_path=output_path,
        conf_threshold=CONF_THRESHOLD
    )

    # Print the results
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    print(f"Total regions detected: {result['num_detections']}")
    print(f"Number of classes: {len(result['detected_diseases'])}")
    print(f"Detected abnormalities: {', '.join(result['detected_diseases'])}")
    print(f"Overlay saved to: {result['overlay_path']}")


if __name__ == '__main__':
    main()