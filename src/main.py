from agent.planner import PlannerModel
from agent.coder import CoderModel
from tools.tools import Tools
from executor.executor import Executor

if __name__ == "__main__":
    planner = PlannerModel()
    code_generator = CoderModel()

    tools = Tools()
    executor = Executor(time_limit_s=60)

    query = """
    Đây là con gì?
    """

    image_path = "/Users/uncpham/Repo/Medical-Assistant/src/static/anh_meo_hai_huoc1.jpg"
    plan = planner.run(query)
    print(f"Plan: {plan}")
    code = code_generator.run(query, image_path)

    print(f"Code: {code}")

    env = tools.as_exec_env()
    result = executor.run(code, env)
    print(result)
    