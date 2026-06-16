"""模型与训练超参数配置。"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """2M 参数 Transformer 配置。

    默认配置约产生 2M 参数（取决于 vocab_size）。
    通过调整 d_model / n_layers 可快速切换到 5M / 10M 档位。
    """

    vocab_size: int = 8192
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0

    # --- 快捷预设 ---
    @classmethod
    def tiny_2m(cls) -> "ModelConfig":
        return cls()

    @classmethod
    def small_5m(cls) -> "ModelConfig":
        return cls(d_model=192, n_layers=8, d_ff=768)

    @classmethod
    def medium_10m(cls) -> "ModelConfig":
        return cls(d_model=256, n_layers=10, d_ff=1024)


@dataclass
class TrainConfig:
    """训练超参数。"""

    # --- 通用 ---
    seed: int = 42
    device: str = "auto"
    dtype: str = "bfloat16"
    compile_model: bool = False

    # --- 阶段 1: 领域预训练 ---
    pretrain_epochs: int = 50
    pretrain_lr: float = 3e-4
    pretrain_min_lr: float = 1e-5
    pretrain_batch_size: int = 16
    pretrain_warmup_steps: int = 100

    # --- 阶段 2: 知识蒸馏 ---
    distill_epochs: int = 5
    distill_lr: float = 4e-4
    distill_min_lr: float = 4e-5
    distill_batch_size: int = 32
    distill_temperature: float = 2.0
    distill_alpha: float = 0.5

    # --- 阶段 3: 指令微调 ---
    finetune_epochs: int = 30
    finetune_lr: float = 5e-5
    finetune_min_lr: float = 1e-6
    finetune_batch_size: int = 16
    finetune_warmup_steps: int = 100

    # --- 正则化 ---
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    label_smoothing: float = 0.1


@dataclass
class PathConfig:
    """文件路径配置。"""

    root: Path = field(default_factory=lambda: Path(__file__).parent)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_text(self) -> Path:
        return self.data_dir / "corpus.txt"

    @property
    def qa_json(self) -> Path:
        return self.data_dir / "qa_pairs.json"

    @property
    def refuse_json(self) -> Path:
        return self.data_dir / "refuse_pairs.json"

    @property
    def tokenizer_prefix(self) -> Path:
        return self.data_dir / "tokenizer"

    @property
    def tokenizer_model(self) -> Path:
        return self.data_dir / "tokenizer.model"

    @property
    def checkpoint_dir(self) -> Path:
        d = self.root / "checkpoints"
        d.mkdir(exist_ok=True)
        return d
