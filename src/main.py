from llm_models.gemini_model import GeminiModel
from tools.tools import Tools
from executor.executor import SafeExecutor

if __name__ == "__main__":
    gemini_model = GeminiModel()
    tools = Tools()
    executor = SafeExecutor(time_limit_s=20)
    
    query = """Hãy tạo ra code để chạy segment_anything tool trên file ảnh '../simple_viper/demo.png'. Đầu ra của bạn chỉ là code python và không bao gồm ```python ```. 
    
    Sử dụng positional arguments (không dùng keyword arguments):
    - segment_anything('../simple_viper/demo.png', [(50.0, 50.0, 200.0, 200.0)])
    
    Ví dụ code:
    result = segment_anything('../simple_viper/demo.png', [(50.0, 50.0, 200.0, 200.0)])
    FINAL_ANSWER = result
    
    Code phải có biến FINAL_ANSWER ở cuối chứa kết quả cuối cùng. Code không được tự định nghĩa hàm và không được import bất cứ thư viện nào."""
    response = gemini_model.run(query)
    print(response)
    env = tools.as_exec_env()
    result = executor.run(response, env)
    print(result)
    