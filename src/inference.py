import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class QASystem:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device_map="auto",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        # Load the fine-tuned weights
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        
        # Ensure model is in float32
        if self.device == "cpu":
            self.model = self.model.float()

    def generate_answer(self, question, language="English"):
        # Format input based on language
        if language == "বাংলা":
            prompt = f"প্রশ্ন: {question}\nউত্তর:"
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode and return
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Answer:" if language == "English" else "উত্তর:")[1].strip() 