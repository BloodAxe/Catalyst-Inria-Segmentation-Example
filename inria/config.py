import sys
from typing import List


@dataclass
class ModelConfig:
    name: str

@dataclass
class OptimizerConfig:
    name: str
    learning_rate:float

@dataclass
class StageConfig:
    epochs: int
    optimizer: OptimizerConfig


@dataclass
class SessionConfig:
    data_dir: str
    batch_size:int
    image_size:int
    train_mode:int

    stages: List[StageConfig]
