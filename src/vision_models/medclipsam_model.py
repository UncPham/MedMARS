import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torchvision
import open_clip
import torch.nn as nn
import warnings
import urllib.request
from PIL import Image
from skimage.util import img_as_float
from skimage import color
import gc
import validators
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from skimage.segmentation import relabel_sequential
from IPython.display import display
import argparse

from src.vision_models.MedCLIP_SAM.saliency_maps.pytorch_grad_cam.utils.image import (
    show_cam_on_image,
)
from src.vision_models.MedCLIP_SAM.saliency_maps.model_loader.clip_loader import (
    load_clip,
)
from src.vision_models.MedCLIP_SAM.saliency_maps.tools.cam import CAMWrapper
from src.vision_models.MedCLIP_SAM.segment_anything.segment_anything import (
    sam_model_registry,
    SamPredictor,
)
from src.vision_models.base_model import BaseModel

warnings.filterwarnings("ignore", category=DeprecationWarning)


class BiomedCLIP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, texts):
        # Getting Image and Text Features

        image_embeddings, text_embeddings, logit_scale = self.model(images, texts)

        # Calculating the Loss
        image_logits = logit_scale * image_embeddings @ text_embeddings.t()
        text_logits = logit_scale * text_embeddings @ image_embeddings.t()

        return image_logits, text_logits


class MedCLIPSAMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.load_model()

    def load_model(self):
        clip_version = "ViT-B/16"  # @param ["RN50x16", "RN50x4", "RN50", "RN101", "ViT-B/32", "ViT-B/16", "hila"]
        cam_version = "gScoreCAM"  # @param ['GradCAM', 'ScoreCAM', 'GracCAM++', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'EigengradCAM', 'LayerCAM', 'HilaCAM', 'GroupCAM', 'SSCAM1', 'SSCAM2', 'RawCAM', 'GradientCAM', 'gScoreCAM']
        self.sam_checkpoint = "/workspace/Medical-Assistant/src/vision_models/MedCLIP_SAM/segment_anything/sam_vit_h_4b8939.pth"  # @param {type:"string"}
        self.model_type = "vit_h"  # @param ["vit_h", "vit_b","vit_l"] {type:"string"}

        topk_channels = 300  # @param {type:"slider", min:1, max:3072, step:1}
        cam_version = cam_version.lower()
        is_transformer = "vit" in clip_version.lower()
        self.model_domain = "biomedclip"

        _, _, _, cam_trans, clip = load_clip(clip_version)
        biomedclip_model, self.preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

        self.clip_model = BiomedCLIP(biomedclip_model)

        target_layer = self.clip_model.model.visual.trunk.blocks[11].norm2

        self.cam_wrapper = CAMWrapper(
            self.clip_model,
            preprocess=self.preprocess,
            target_layers=[target_layer],
            tokenizer=clip.tokenize,
            drop=True,
            cam_version=cam_version,
            topk=topk_channels,
            channels=None,
            is_transformer=is_transformer,
            cam_trans=cam_trans,
            model_domain=self.model_domain,
        )

        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)

        self.predictor = SamPredictor(self.sam)

    def get_visualization(self, img, prompt):
        # encode image and prompt
        raw_size = img.size
        input_img = self.preprocess(img).unsqueeze(0).cuda()
        text_token = self.tokenizer(prompt).cuda()
        # get cam for prompt and overlay on input image
        cam = self.cam_wrapper.getCAM(
            input_img, text_token, raw_size, 0, model_domain=self.model_domain
        )

        float_img = img_as_float(img)
        if len(float_img.shape) == 2:
            float_img = color.gray2rgb(float_img)
        cam_img = show_cam_on_image(float_img, cam, use_rgb=True)
        saliency_map = cam_img.copy()
        cam_img = Image.fromarray(cam_img)
        cat_img = Image.new("RGB", (raw_size[0] * 2, raw_size[1]))
        cat_img.paste(img, (0, 0))
        cat_img.paste(cam_img, (raw_size[0], 0))
        return cat_img, saliency_map

    def get_saliency_maps(self, image_path, prompt):
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        try:
            img = Image.open(image_path)
        except Exception as e:
            raise Exception(f"image is not valid. Error: {e}")

        visualization, saliency_map = self.get_visualization(img, prompt)
        return visualization, saliency_map

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def postprocess_crf(
        self,
        image_path,
        saliency_map,
        M: int = 2,
        tau: float = 1.4,
        gaussian_sxy: int = 25,
        gaussian_rgb: int = 5,
        bilateral_sxy: int = 25,
        bilateral_srgb: int = 5,
    ):
        img = cv2.imread(image_path, 1)
        annos = np.squeeze(saliency_map[:, :, 0])
        labels = relabel_sequential(annos)[0].flatten()
        output = image_path.replace(".png", "_crf.png")

        # Setup the CRF model
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

        anno_norm = annos / 255.0
        n_energy = -np.log((1.0 - anno_norm + 1e-8)) / (
            tau * self.sigmoid(1 - anno_norm)
        )
        p_energy = -np.log(anno_norm + 1e-8) / (tau * self.sigmoid(anno_norm))

        U = np.zeros((M, img.shape[0] * img.shape[1]), dtype="float32")
        U[0, :] = n_energy.flatten()
        U[1, :] = p_energy.flatten()

        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=gaussian_sxy, compat=3)
        d.addPairwiseBilateral(
            sxy=bilateral_sxy, srgb=bilateral_srgb, rgbim=img, compat=5
        )

        # Do the inference
        Q = d.inference(1)
        map = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

        # Save the output as image
        map *= 255
        cv2.imwrite(output, map.astype("uint8"))

        return map.astype("uint8")

    def scoremap2bbox(self, scoremap, threshold, multi_contour_eval=False):
        _CONTOUR_INDEX = 1 if cv2.__version__.split(".")[0] == "3" else 0
        height, width = scoremap.shape
        scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY,
        )
        contours = cv2.findContours(
            image=thr_gray_heatmap, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, width, height]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return estimated_boxes, contours

    def __call__(self, image_path, prompt):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        visualization, saliency_map = self.get_saliency_maps(image_path, prompt)

        mask = self.postprocess_crf(image_path, saliency_map)

        boxes, _ = self.scoremap2bbox(mask, 0, multi_contour_eval=False)
point_coords
        self.predictor.set_image(img)
        boxes = np.array(boxes)

        masks, _, _ = self.predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=False
        )

        output = image_path.replace(".png", "_sam.png")
        mask_sam = np.squeeze(masks * 255).astype("uint8")
        cv2.imwrite(output, mask_sam)
        Image.fromarray(mask_sam).save(output)
        print(output)
        print(mask_sam.shape)
        display(mask_sam)
        return


if __name__ == "__main__":
    model = MedCLIPSAMModel()
    image_path = "/workspace/Medical-Assistant/src/test/MedCLIP-SAM/assets/example.png"
    prompt = "brain tumor"
    model(image_path, prompt)
