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
        Perform segmentation on a chest X-ray image and save results.

        Args:
            image_path: Path to the input chest X-ray image

        Returns:
            dict: Dictionary containing paths to saved images:
                - 'overlay_path': Path to the segmentation overlay image
                - 'RL_mask_path': Path to the right lung mask image
                - 'LL_mask_path': Path to the left lung mask image
                - 'H_mask_path': Path to the heart mask image
        """
        # Perform segmentation - returns (outseg, file_list)
        # The segment function saves files to tmp/ directory
        outseg, file_list = segment(image_path)

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Get base filename from input image
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Get path to tmp directory
        hybridgnet_dir = os.path.join(os.path.dirname(__file__), "Chest_x_ray_HybridGNet_Segmentation")
        tmp_dir = os.path.join(hybridgnet_dir, "tmp")

        # Copy files from tmp directory to output directory with new names
        overlay_filename = f'{base_filename}_segmentation_lungs_heart_overlay.png'
        overlay_path = os.path.join(self.output_dir, overlay_filename)
        shutil.copy(os.path.join(tmp_dir, "overlap_segmentation.png"), overlay_path)

        rl_mask_path = os.path.join(self.output_dir, f'{base_filename}_RL_mask.png')
        ll_mask_path = os.path.join(self.output_dir, f'{base_filename}_LL_mask.png')
        h_mask_path = os.path.join(self.output_dir, f'{base_filename}_H_mask.png')

        shutil.copy(os.path.join(tmp_dir, "RL_mask.png"), rl_mask_path)
        shutil.copy(os.path.join(tmp_dir, "LL_mask.png"), ll_mask_path)
        shutil.copy(os.path.join(tmp_dir, "H_mask.png"), h_mask_path)

        # Return only the paths
        result = {
            'overlay_path': overlay_path,
            'RL_mask_path': rl_mask_path,
            'LL_mask_path': ll_mask_path,
            'H_mask_path': h_mask_path
        }

        return result
    
if __name__ == "__main__":
    model = CXRHybridGNetSegmentationModel()
    image_path = '/Users/uncpham/Repo/Medical-Assistant/src/data/vqa_rad/images/img_0.jpg'
    result = model(image_path)
    print(f"Overlay path: {result['overlay_path']}")
    print(f"Right lung mask path: {result['RL_mask_path']}")
    print(f"Left lung mask path: {result['LL_mask_path']}")
    print(f"Heart mask path: {result['H_mask_path']}")
