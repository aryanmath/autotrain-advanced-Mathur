from dataclasses import dataclass
from typing import Optional

from autotrain.trainers.common import TrainingParams


@dataclass
class AutomaticSpeechRecognitionParams(TrainingParams):
    max_duration: float = 30.0
    sampling_rate: int = 16000
    audio_column: str = "audio"
    text_column: str = "text"
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    eval_strategy: str = "epoch"
    save_total_limit: int = 1
    auto_find_batch_size: bool = False
    logging_steps: int = -1