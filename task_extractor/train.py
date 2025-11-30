import os, json, time, argparse, logging
from pathlib import Path
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from task_extractor.models import TaskExtractionTransformer
from task_extractor.data import TaskTokenizer, TaskExtractionDataset, DataCollator, create_data_loaders, BIO_TAGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(_name_)

@dataclass
class TrainingConfig:
    data_path: str = "training_data_100k.jsonl"
    output_dir: str = "checkpoints"
    max_seq_len: int = 512
    val_split: float = 0.1
    vocab_size: int = 30000
    d_model: int = 1024
    num_heads: int = 16
    num_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    use_crf: bool = True
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 3000
    max_epochs: int = 30
    grad_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    use_amp: bool = True
    save_every_n_epochs: int = 1
    eval_every_n_steps: int = 1000
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 0.001
    
    def to_dict(self): return asdict(self)
    @classmethod
    def from_json(cls, path):
        with open(path) as f: return cls(**json.load(f))

class Trainer:
    def _init_(self, model, train_loader, val_loader, config, tokenizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=config.learning_rate, total_steps=len(train_loader)*config.max_epochs, pct_start=0.1)
        self.use_amp = config.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        os.makedirs(config.output_dir, exist_ok=True)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], bio_labels=batch['bio_labels'], assignee_labels=batch['assignee_labels'], priority_labels=batch['priority_labels'], deadline_labels=batch['deadline_labels'])
                loss = outputs['loss'] / self.config.grad_accumulation_steps
            
            if self.scaler: self.scaler.scale(loss).backward()
            else: loss.backward()
            
            if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.config.grad_accumulation_steps
            if self.global_step % self.config.eval_every_n_steps == 0:
                val_metrics = self.evaluate()
                logger.info(f"Step {self.global_step} | Val Loss: {val_metrics['loss']:.4f}")
                self.model.train()
        return {'loss': total_loss / len(self.train_loader)}

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, total_bio_correct, total_bio_total = 0.0, 0, 0
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], bio_labels=batch['bio_labels'], assignee_labels=batch['assignee_labels'], priority_labels=batch['priority_labels'], deadline_labels=batch['deadline_labels'])
            total_loss += outputs['loss'].item()
            mask = batch['bio_labels'] != -100
            total_bio_correct += ((outputs['bio_logits'].argmax(dim=-1) == batch['bio_labels']) & mask).sum().item()
            total_bio_total += mask.sum().item()
        return {'loss': total_loss / len(self.val_loader), 'bio_accuracy': total_bio_correct / max(total_bio_total, 1)}

    def train(self):
        logger.info("Starting training...")
        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            self.training_history.append({'epoch': epoch, 'train': train_metrics, 'val': val_metrics})
            logger.info(f"Epoch {epoch} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | BIO Acc: {val_metrics['bio_accuracy']:.4f}")
            
            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, val_metrics['loss'])
            
            if val_metrics['loss'] < self.best_val_loss - self.config.early_stopping_min_delta:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics['loss'], is_best=True)
            else:
                self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience: break
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(), 'val_loss': val_loss, 'config': self.config.to_dict()}
        torch.save(checkpoint, os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch}.pt'))
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, 'best_model.pt'))
            self.tokenizer.save(os.path.join(self.config.output_dir, 'tokenizer.json'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='meeting_datasets_5000.jsonl')
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    config = TrainingConfig(data_path=args.data_path, output_dir=args.output_dir, batch_size=args.batch_size)
    tokenizer = TaskTokenizer(vocab_size=config.vocab_size)
    
    # Simple training simulation if data doesn't exist
    if not os.path.exists(config.data_path):
        print("Training data not found. Please run generate_training_data.py first.")
        return

    train_loader, val_loader = create_data_loaders(config.data_path, tokenizer, batch_size=config.batch_size)
    model = TaskExtractionTransformer.from_config(config.to_dict())
    trainer = Trainer(model, train_loader, val_loader, config, tokenizer)
    trainer.train()

if _name_ == '_main_':
    main()