import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from transformers import CLIPModel, CLIPProcessor
from typing import List
from PIL import Image

from src.vision_models.base_model import BaseModel


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

if __name__ == "__main__":
    model = ClipModel("/Users/uncpham/Repo/Medical-Assistant/src/checkpoint/clip-vit-base-patch32")
    image_path = "/Users/uncpham/Repo/Medical-Assistant/src/static/anh_meo_hai_huoc1.jpg"
    labels = ["cat", "dog"]
    outputs = model.forward(image_path, labels)
    
    print("=== CLIP Model Output Analysis ===")
    print(f"Output type: {type(outputs)}")
    print(f"Available attr`ibutes/keys: {dir(outputs)}")
    print()
    
    # Display specific fields
    if hasattr(outputs, 'logits_per_image'):
        print(f"logits_per_image shape: {outputs.logits_per_image.shape}")
        print(f"logits_per_image values: {outputs.logits_per_image}")
        print()
    
    if hasattr(outputs, 'logits_per_text'):
        print(f"logits_per_text shape: {outputs.logits_per_text.shape}")
        print(f"logits_per_text values: {outputs.logits_per_text}")
        print()
    
    if hasattr(outputs, 'text_embeds'):
        print(f"text_embeds shape: {outputs.text_embeds.shape}")
        print(f"text_embeds (first 5 values): {outputs.text_embeds[0][:5]}")
        print()
    
    if hasattr(outputs, 'image_embeds'):
        print(f"image_embeds shape: {outputs.image_embeds.shape}")
        print(f"image_embeds (first 5 values): {outputs.image_embeds[0][:5]}")
        print()
    
    # Calculate probabilities
    probs = outputs.logits_per_image.softmax(dim=-1)
    print("=== Probability Analysis ===")
    for i, label in enumerate(labels):
        print(f"{label}: {probs[0][i].item():.4f} ({probs[0][i].item()*100:.2f}%)")
    
    print(f"\nMost likely label: {labels[probs.argmax().item()]}")
    print(f"Confidence: {probs.max().item():.4f} ({probs.max().item()*100:.2f}%)")
