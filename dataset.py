"""数据集类：预训练、QA 指令微调、蒸馏。"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from config import PathConfig


class PretrainDataset(Dataset):
    """因果语言建模预训练数据集。

    将原始文本切分为固定长度的 token 序列，
    用 next-token prediction 目标训练。
    """

    def __init__(
        self,
        corpus_path: Path,
        tokenizer: Tokenizer,
        max_len: int = 512,
    ):
        text = corpus_path.read_text(encoding="utf-8")
        # 不加 BOS/EOS，整段文本连续编码
        tokenizer.no_padding()
        tokenizer.no_truncation()
        encoded = tokenizer.encode(text)
        self.token_ids = encoded.ids

        self.max_len = max_len
        # 切分为不重叠的块
        n = len(self.token_ids)
        self.chunks = [
            self.token_ids[i : i + max_len + 1]
            for i in range(0, n - max_len - 1, max_len)
        ]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}


class InstructDataset(Dataset):
    """指令微调数据集。

    支持两种格式：
    1. QA 对：{"question": "...", "answer": "..."}
    2. 拒答对：{"question": "...", "answer": "抱歉..."}

    自动包装为 ChatML 格式。
    """

    SYSTEM_PROMPT = (
        "你是一个专业的角色助手，只回答与你的知识领域相关的问题。"
        "对于领域外的问题，请礼貌地拒绝。"
    )

    def __init__(
        self,
        data_paths: list[Path],
        tokenizer: Tokenizer,
        max_len: int = 512,
        system_prompt: str | None = None,
        max_refuse: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

        self.samples: list[dict] = []
        self.is_refuse: list[bool] = []
        for p in data_paths:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    items = json.load(f)
                is_ref = "refuse" in p.stem
                if is_ref and max_refuse and len(items) > max_refuse:
                    random.shuffle(items)
                    items = items[:max_refuse]
                self.samples.extend(items)
                self.is_refuse.extend([is_ref] * len(items))

        combined = list(zip(self.samples, self.is_refuse))
        random.shuffle(combined)
        self.samples = [c[0] for c in combined]
        self.is_refuse = [c[1] for c in combined]

    def _format_chat(self, question: str, answer: str) -> str:
        return (
            f"<|system|>{self.system_prompt}<|end|>"
            f"<|user|>{question}<|end|>"
            f"<|assistant|>{answer}<|end|>"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        text = self._format_chat(item["question"], item["answer"])

        encoded = self.tokenizer.encode(text)
        ids = encoded.ids[: self.max_len]

        # padding
        pad_len = self.max_len - len(ids)
        input_ids = ids + [0] * pad_len  # 0 = <pad>
        labels = ids[1:] + [0] * (pad_len + 1)

        # 只在 assistant 回复部分计算 loss
        assistant_token = self.tokenizer.token_to_id("<|assistant|>")
        found = False
        for i, tid in enumerate(input_ids):
            if tid == assistant_token:
                found = True
            if not found:
                labels[i] = -100
            if tid == 0:
                labels[i] = -100

        input_ids = torch.tensor(input_ids[:self.max_len], dtype=torch.long)
        labels = torch.tensor(labels[:self.max_len], dtype=torch.long)
        # QA 样本权重 3.0，拒答样本权重 0.3
        weight = 0.3 if self.is_refuse[idx] else 3.0
        return {
            "input_ids": input_ids,
            "labels": labels,
            "weight": torch.tensor(weight, dtype=torch.float),
        }


class DistillDataset(Dataset):
    """蒸馏数据集。

    存储 teacher 模型的 soft label（logits 或 top-k 概率）。
    格式：{"input_ids": [...], "teacher_logits": [...]}
    """

    def __init__(self, data_path: Path, max_len: int = 512):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        input_ids = torch.tensor(
            item["input_ids"][: self.max_len], dtype=torch.long
        )
        teacher_logits = torch.tensor(
            item["teacher_logits"], dtype=torch.float
        )
        labels = torch.tensor(
            item.get("labels", item["input_ids"][1:] + [0])[: self.max_len],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "teacher_logits": teacher_logits,
        }
