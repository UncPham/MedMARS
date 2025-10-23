import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from PIL import Image
from typing import Union
import numpy as np


from src.agent.planner import PlannerModel
from src.agent.coder import CoderModel
from src.agent.reporter import ReporterModel
# from src.tools.tools import Tools
from executor.executor import Executor
from src.image_patch import ImagePatch


class MedMARS:
    def __init__(self, max_rounds: int = 3):
        self.planner = PlannerModel()
        self.code_generator = CoderModel()
        self.reporter = ReporterModel()
        # self.tools = Tools()
        self.executor = Executor(time_limit_s=60)

        self.thought = None
        self.plan = None
        self.code = None

        self.max_rounds = max_rounds

    def run(self, query: str, image: Union[str, Path, Image.Image]):
        if isinstance(image, (str, Path)):
            image_path = str(Path(image))
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image, Image.Image):
            image_path = 'imgs/query_image.png'
            image.save(image_path)
            image = np.array(image)
        else:
            raise ValueError("image_path must be a str, Path, or PIL.Image")

        self.thought, self.plan = self.planner(query=query, image_path=image_path)
        print(5*'-',"Thought", 5*'-', '\n' + self.thought)
        print(5*'-',"Plan", 5*'-', '\n' + self.plan)

        self.code = self.code_generator(self.plan)
        print(5*'-',"Code", 5*'-', '\n' + self.code)

        print("Executing code...")
        exec_globals = globals().copy()
        exec_globals['ImagePatch'] = ImagePatch
        exec(self.code, exec_globals)
        result = None
        try:
            execute_command = exec_globals.get('execute_command')
            if execute_command is None:
                raise ValueError("execute_command function not found in generated code")
            out = execute_command(image_path)
            result = out.copy() if hasattr(out, 'copy') else out
        except Exception as e:
            out = str(e)
            result = None
            print("Error:", out)

        # Generate answer and explanation from output
        print("Generating answer...")
        response = self.reporter(query, out, self.plan) if out else {
            "answer": "Error in execution",
            "explanation": str(result),
            "report": str(result)  # Backward compatibility
        }

        # env = self.tools.as_exec_env()
        # result = self.executor.run(self.code, env)
        return out, result, response


if __name__ == "__main__":
    medmars = MedMARS()
    image_path = '/home/xuananh/work_1/chien/Medical-Assistant/src/data/vqa_rad/images/img_32.jpg'
    query = "is this supratentorial or infratentorial?"
    out, result, response = medmars.run(query, image_path)
    print(5*'-',"Output", 5*'-', '\n' + str(out))
    print(5*'-',"Result", 5*'-', '\n' + str(result))
    print(5*'-',"Response", 5*'-')
    print("Answer:", response.get('answer'))
    print("\nExplanation:", response.get('explanation'))
    # Backward compatibility
    if not response.get('explanation') and response.get('report'):
        print("\nReport:", response.get('report'))