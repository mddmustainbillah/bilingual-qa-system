from trl import DPOTrainer

def train_with_dpo(model, tokenizer, dataset):
    dpo_trainer = DPOTrainer(
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            learning_rate=5e-5,
            num_train_epochs=1,
            output_dir="dpo_results"
        ),
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    dpo_trainer.train()
    return model 