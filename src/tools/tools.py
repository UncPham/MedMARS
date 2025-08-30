from vision_models.segment_anything import SegmentAnything
from vision_models.clip_model import ClipModel

class Tools:
    def __init__(self):
        self.segment_anything = SegmentAnything(model_path="facebook/sam-vit-huge")
        self.clip_model = ClipModel(model_path="openai/clip-vit-base-patch32")

    def as_exec_env(self):
        return {
            "segment_anything": self.segment_anything.forward,
            "clip_model": self.clip_model.forward
        }