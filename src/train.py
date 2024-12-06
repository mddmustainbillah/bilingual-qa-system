import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
import numpy as np

def load_model_and_tokenizer():
    # Load base model (Llama 2)
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_model_for_training():
    model, tokenizer = load_model_and_tokenizer()
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def prepare_dataset():
    # Load an even smaller subset of SQuAD v2 dataset
    dataset = load_dataset("squad_v2", split="train[:100]")  # Reduced from 1000 to 100 examples
    
    def format_dataset(example):
        """Format the dataset into instruction format"""
        # Add a check for the key or provide a default value
        is_impossible = example.get('is_impossible', False)
        
        if is_impossible:
            answer = "I cannot find the answer to this question in the given context."
        else:
            answer = example['answers']['text'][0]
            
        # Format into instruction format
        context = example['context']
        question = example['question']
        
        # Create prompt in Llama2 format
        formatted_prompt = f"""<s>[INST] Context: {context}

Question: {question} [/INST]

{answer}</s>"""
        
        return {"text": formatted_prompt}
    
    # Format the dataset
    formatted_dataset = dataset.map(format_dataset)
    
    # Convert to the format expected by the trainer
    formatted_dataset = formatted_dataset.remove_columns([
        'id', 'title', 'context', 'question', 'answers'
    ])
    
    return formatted_dataset

def train():
    model, tokenizer = prepare_model_for_training()
    
    # Load and prepare the dataset
    dataset = prepare_dataset()
    
    # Tokenize the dataset
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        # Add labels for the loss calculation
        result["labels"] = result["input_ids"].clone()
        return result
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments with optimized settings for CPU
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,            # Kept at 1 epoch
        per_device_train_batch_size=1, # Kept small for CPU
        gradient_accumulation_steps=4,  # Reduced from 8 to 4
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        save_total_limit=1,            # Reduced from 3 to 1
        logging_steps=10,              # Reduced from 50 to 10
        save_strategy="epoch",         # Changed from "steps" to "epoch"
        warmup_steps=10,               # Reduced from 50 to 10
        optim="adamw_torch",
        evaluation_strategy="no",      # Added to disable evaluation
        report_to="none",             # Disable wandb or other reporting
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save the model
    trainer.save_model("./models/qa_model")

if __name__ == "__main__":
    train() 