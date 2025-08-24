from transformers import AutoTokenizer
import transformers
import torch

class CodeLLamaModel():
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def forward(self, prompt: str):
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            temperature=0.1,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=200,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")

if __name__ == "__main__":
    model = CodeLLamaModel("codellama/CodeLlama-7b-hf")
    model.forward("# Function to calculate the factorial of a number\ndef factorial(n):")