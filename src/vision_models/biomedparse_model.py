import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.vision_models.BiomedParse.modeling.BaseModel import BaseModel
from src.vision_models.BiomedParse.modeling import build_model
from src.vision_models.BiomedParse.utilities.distributed import init_distributed # changed from utils
from src.vision_models.BiomedParse.utilities.arguments import load_opt_from_config_files
from src.vision_models.BiomedParse.inference_utils.output_processing import check_mask_stats
from src.vision_models.BiomedParse.utilities.constants import BIOMED_CLASSES
from src.vision_models.BiomedParse.inference_utils.inference import interactive_infer_image

# from src.vision_models.base_model import BaseModel

class BiomedParseModel():
    def __init__(self):
        super().__init__()
        self.load_model()
    
    def load_model(self):
        conf_files = os.path.join(os.path.dirname(__file__), "BiomedParse/configs/biomedparse_inference.yaml")
        opt = load_opt_from_config_files([conf_files])
        opt = init_distributed(opt)

        model_file = os.path.join(os.path.dirname(__file__), "..", "checkpoint/biomedparse_v1.pt")

        self.model = BaseModel(opt, build_model(opt)).from_pretrained(model_file).eval().cuda()
        with torch.no_grad():
            self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

    def __call__(self, image_path: str, prompts: list[str]):
        image = Image.open(image_path).convert('RGB')
        pred_mask = interactive_infer_image(self.model, image, prompts)
        return pred_mask

    @staticmethod
    def overlay_masks(image, masks, colors):
        overlay = image.copy()
        overlay = np.array(overlay, dtype=np.uint8)
        for mask, color in zip(masks, colors):
            overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
        return Image.fromarray(overlay)

if __name__ == "__main__":
    model = BiomedParseModel()
    image_path = '/work/chien/Medical-Assistant/src/static/img_0.jpg'
    prompts = ['one heart', 'one left lung', 'one right lung']
    pred_mask = model(image_path, prompts)

    for i, pred in enumerate(pred_mask):
        mask_image = Image.fromarray(((pred * 225)).astype(np.uint8))
        output_path = f'/work/chien/Medical-Assistant/src/static/img_0_{prompts[i].replace(" ", "_")}_pred.png'
        mask_image.save(output_path)
        print(f'Saved prediction mask for {prompts[i]} to {output_path}')

    image = Image.open(image_path).convert('RGB')
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    overlay_image = model.overlay_masks(image, [1*(pred_mask[i] > 0.5) for i in range(len(prompts))], colors)
    overlay_image.save('/work/chien/Medical-Assistant/src/static/img_0_overlay.png')





