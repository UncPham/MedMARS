import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_models.biomedclip_model import BioMedClipModel
from vision_models.groundingdino_model import GroundingDINOModel
from vision_models.midas_model import MiDaSModel
from vision_models.medsam_model import MedSAMModel

class Tools:
    def __init__(self):
        self.clip_model = BioMedClipModel()
        self.glip_model = GroundingDINOModel()
        self.midas_model = MiDaSModel()
        self.medsam_model = MedSAMModel()
        self.segment_anything = None  # Placeholder for Segment Anything Model

    def as_exec_env(self):
        return {
            "biomedclip_model": self.clip_model.__call__,
            "groundingdino_model": self.glip_model.__call__,
            "midas_model": self.midas_model.__call__,
            "medsam_model": self.medsam_model.__call__,
        }