import torch
from transformers import CLIPModel, CLIPProcessor
from typing import List
from PIL import Image

from vision_models.base_model import BaseModel


class ClipModel(BaseModel):
    def __init__(self, model_path: str):
        super().__init__()
        self.load_model(model_path)

    def forward(self, image_path: str, labels: List[str]):
        """
        This function uses the CLIP model to compute similarity between an image and text labels,
        helping to find the most suitable label for the input image.

        Input:
        - image_path (str): Path to the image file to be processed.
        - labels (List[str]): List of text labels to compare with the image.

        Output:
        - outputs (dict): Dictionary containing CLIP model results with the following keys:
            - logits_per_image: Tensor of shape (1, num_labels) containing similarity scores between the image and each text label
            - logits_per_text: Tensor of shape (num_labels, 1) containing similarity scores between each text label and the image
            - text_embeds: Normalized embeddings for the text labels
            - image_embeds: Normalized embeddings for the image
        """
        raw_image = Image.open(image_path).convert("RGB")
        # Use CLIP to find the most suitable patch
        inputs = self.clip_processor(
            text=labels, images=raw_image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs

    def load_model(self, model_path: str):
        self.clip_model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_path)
