import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class ModelConfig:
    """Model architecture configuration - ~100M parameters."""
    vocab_size: int = 30000
    d_model: int = 1024
    num_heads: int = 16
    num_layers: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    num_assignees: int = 5
    num_priorities: int = 4
    num_bio_tags: int = 9
    dropout: float = 0.1
    use_crf: bool = True
    pre_norm: bool = True


@dataclass
class TrainingConfig:
    """Training configuration for 100M model."""
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_epochs: int = 30
    grad_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    use_amp: bool = True
    early_stopping_patience: int = 7
    eval_every_n_steps: int = 500


@dataclass
class TeamConfig:
    """Team member configuration."""
    members: Dict[str, List[str]] = field(default_factory=lambda: {
        'mohit': ['backend', 'api', 'database', 'performance', 'caching'],
        'lata': ['frontend', 'ui', 'design', 'css', 'responsive'],
        'arjun': ['testing', 'qa', 'automation', 'quality', 'unit tests'],
        'sakshi': ['devops', 'deployment', 'infrastructure', 'monitoring', 'documentation']
    })


@dataclass
class DeepgramConfig:
    """Deepgram API configuration."""
    api_key: Optional[str] = None
    model: str = "nova-2"
    language: str = "en-US"
    punctuate: bool = True
    diarize: bool = True
    smart_format: bool = True
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get('DEEPGRAM_API_KEY')


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class Config:
    """Complete application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    deepgram: DeepgramConfig = field(default_factory=DeepgramConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Paths
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    data_path: str = "meeting_datasets_5000.jsonl"
    output_dir: str = "checkpoints"
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        with open(path) as f:
            data = json.load(f)
        
        return cls(
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            team=TeamConfig(**data.get('team', {})),
            deepgram=DeepgramConfig(**data.get('deepgram', {})),
            api=APIConfig(**data.get('api', {})),
            model_path=data.get('model_path'),
            tokenizer_path=data.get('tokenizer_path'),
            data_path=data.get('data_path', 'meeting_datasets_5000.jsonl'),
            output_dir=data.get('output_dir', 'checkpoints')
        )
    
    def to_json(self, path: str):
        data = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'team': {'members': self.team.members},
            'deepgram': {k: v for k, v in self.deepgram.__dict__.items() if k != 'api_key'},
            'api': self.api.__dict__,
            'model_path': self.model_path,
            'tokenizer_path': self.tokenizer_path,
            'data_path': self.data_path,
            'output_dir': self.output_dir
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_env(cls) -> 'Config':
        config = cls()
        if os.environ.get('MODEL_PATH'): config.model_path = os.en