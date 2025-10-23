import os
import sys
import cv2
import torch
import shutil

# Add paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.join(os.path.dirname(__file__), "Chest_x_ray_HybridGNet_Segmentation"))

from src.vision_models.Chest_x_ray_HybridGNet_Segmentation.app import loadModel, segment
from src.constants.env import STATIC_FOLDER
from src.vision_models.base_model import BaseModel

class CXRHybridGNetSegmentationModel(BaseModel):
    def __init__(self, output_dir: str = None):
        """
        Initialize the CXR HybridGNet Segmentation Model.

        Args:
            output_dir: Directory to save segmentation results. If None, results won't be saved.
        """
        super().__init__()  # Initialize BaseModel to set self.device
        self.model = loadModel(self.device)
        self.output_dir = output_dir if output_dir is not None else STATIC_FOLDER

    def __call__(self, image_path: str):
        """
        Perform segmentation on a chest X-ray image and optionally save results.

        Args:
            image_path: Path to the input chest X-ray image

        Returns:
            dict: Dictionary containing:
                - 'output_seg': The segmentation overlay (numpy array)
                - 'saved_files': List of saved file paths (if output_dir is set)
                - 'overlay_path': Path to the saved overlay image (if output_dir is set)
        """
        # Perform segmentation - returns numpy arrays
        seg_to_save, RL_mask, LL_mask, H_mask = segment(image_path)

        result = {
            
            'overlay': {
                'image': seg_to_save,
                'path': ,
            },
            'RL_mask': RL_mask,
            'LL_mask': LL_mask,
            'H_mask': H_mask,
            'saved_files': []
        }

        # Save results if output_dir is specified
        if self.output_dir is not None:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)

            # Get base filename from input image
            base_filename = os.path.splitext(os.path.basename(image_path))[0]

            # Save overlay image
            overlay_filename = f'{base_filename}_segmentation_overlay.png'
            overlay_path = os.path.join(self.output_dir, overlay_filename)
            cv2.imwrite(overlay_path, cv2.cvtColor(seg_to_save, cv2.COLOR_RGB2BGR))
            result['saved_files'].append(overlay_path)

            # Save mask images
            rl_mask_path = os.path.join(self.output_dir, f'{base_filename}_RL_mask.png')
            ll_mask_path = os.path.join(self.output_dir, f'{base_filename}_LL_mask.png')
            h_mask_path = os.path.join(self.output_dir, f'{base_filename}_H_mask.png')

            cv2.imwrite(rl_mask_path, RL_mask)
            cv2.imwrite(ll_mask_path, LL_mask)
            cv2.imwrite(h_mask_path, H_mask)

            result['saved_files'].extend([rl_mask_path, ll_mask_path, h_mask_path])

        return result
    
if __name__ == "__main__":
    model = CXRHybridGNetSegmentationModel()
    image_path = '/home/xuananh/work_1/chien/Medical-Assistant/src/data/vqa_rad/images/img_0.jpg'
    result = model(image_path)
    print(f"Segmentation overlay shape: {result['seg_overlay'].shape}")
    print(f"Right lung mask shape: {result['RL_mask'].shape}")
    print(f"Left lung mask shape: {result['LL_mask'].shape}")
    print(f"Heart mask shape: {result['H_mask'].shape}")
    print(f"Saved files: {result['saved_files']}")
