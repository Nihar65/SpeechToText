import os, json
import torch
from task_extractor.models import TaskExtractionTransformer
from task_extractor.data import TaskTokenizer, create_data_loaders
from task_extractor.train import Trainer, TrainingConfig

def main():
    if not torch.cuda.is_available(): print("Warning: No GPU detected")
    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Dummy data generation for testing
    if not os.path.exists(config.data_path):
        os.system("python generate_training_data.py")
        
    tokenizer = TaskTokenizer(vocab_size=config.vocab_size)
    # Mock training tokenizer on existing data would go here
    
    train_loader, val_loader = create_data_loaders(config.data_path, tokenizer, batch_size=config.batch_size)
    model = TaskExtractionTransformer.from_config(config.to_dict())
    trainer = Trainer(model, train_loader, val_loader, config, tokenizer)
    trainer.train()

if _name_ == '_main_':
    main()