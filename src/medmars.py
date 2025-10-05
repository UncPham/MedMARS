import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from PIL import Image
from typing import Union
import numpy as np

from agent.planner import PlannerModel
from agent.coder import CoderModel
from tools.tools import Tools
from executor.executor import Executor


class MedMARS:
    def __init__(self, max_rounds: int = 3):
        self.planner = PlannerModel()
        self.code_generator = CoderModel()
        self.tools = Tools()
        self.executor = Executor(time_limit_s=60)

        self.plan = None
        self.code = None

        self.max_rounds = max_rounds

    async def run(self, query: str, image: Union[str, Path, Image.Image]):
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image, Image.Image):
            image = np.array(image)
            image_path = 'imgs/query_image.png'
            image.save(image_path)
        else:
            raise ValueError("image_path must be a str, Path, or PIL.Image")
                      
        self.plan = self.planner(query)
        print(f"Plan: {self.plan}")

        self.code = self.code_generator(self.plan, image_path)
        print(f"Code: {self.code}")

        env = self.tools.as_exec_env()
        result = self.executor.run(self.code, env)
        return result
