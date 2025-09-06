import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_models.segment_anything import SegmentAnything
from vision_models.clip_model import ClipModel
from vision_models.groundingdino_model import GLIPModel

class Tools:
    def __init__(self):
        self.segment_anything = SegmentAnything(model_path="facebook/sam-vit-huge")
        self.clip_model = ClipModel(model_path="openai/clip-vit-base-patch32")
        self.glip_model = GLIPModel(model_path=os.path.join(os.path.dirname(__file__), "..", "checkpoint"))

    def as_exec_env(self):
        return {
            "segment_anything": self.segment_anything.forward,
            "clip_model": self.clip_model.forward,
            "glip_model": self.glip_model.forward
        }