import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from typing import List
from PIL import Image

from src.vision_models.base_model import BaseModel


class BioMedClipModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.load_model()

    def __call__(self, images_path: List[str], labels: List[str]) -> dict:
        images = torch.stack(
            [self.preprocess(Image.open((img))) for img in images_path]
        ).to(self.device)

        texts = self.tokenizer(
            [l for l in labels], context_length=self.context_length
        ).to(self.device)
        
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(images, texts)

            logits = (
                (logit_scale * image_features @ text_features.t())
                .detach()
                .softmax(dim=-1)
            )

            sorted_indices = torch.argsort(logits, dim=-1, descending=True)
            logits = logits.cpu().numpy()
            sorted_indices = sorted_indices.cpu().numpy()

        top_k = -1
        results = {}

        for i, img in enumerate(images_path):
            # Create dictionary for this image with label percentages
            image_results = {}
            top_k = len(labels) if top_k == -1 else top_k
            for j in range(top_k):
                jth_index = sorted_indices[i][j]
                percentage = logits[i][jth_index] * 100  # Convert to percentage
                image_results[labels[jth_index]] = round(percentage, 2)
            
            results[f"image_{img.split('/')[-1]}"] = image_results

        return results

    def load_model(self):
        # Load the model and config files from the Hugging Face Hub
        self.model, self.preprocess = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.tokenizer = get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.model.to(self.device)
        self.model.eval()
        self.context_length = 256


# if __name__ == "__main__":
#     model = BioMedClipModel()
#     image_path = [
#         "/Users/uncpham/Repo/Medical-Assistant/src/data/vqa_rad/images/img_0.jpg"
#     ]
#     text = [
#         "adenocarcinoma histopathology",
#         "brain MRI",
#         "covid line chart",
#         "squamous cell carcinoma histopathology",
#         "immunohistochemistry histopathology",
#         "bone X-ray",
#         "chest X-ray",
#         "pie chart",
#         "hematoxylin and eosin histopathology",
#     ]
#     outputs = model(image_path, text)
#     print(outputs)
