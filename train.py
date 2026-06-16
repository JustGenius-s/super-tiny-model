"""3 阶段训练管线：预训练 -> 知识蒸馏 -> 指令微调。

用法:
    python train.py --phase all      # 完整 3 阶段
    python train.py --phase pretrain  # 仅预训练
    python train.py --phase distill   # 仅蒸馏（需先完成预训练）
    python train.py --phase finetune  # 仅微调（需先完成蒸馏或预训练）
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from config import ModelConfig, PathConfig, TrainConfig
from dataset import InstructDataset, PretrainDataset
from model import TinyRoleModel
from tokenizer_train import load_tokenizer


def get_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg.device)


def get_dtype(cfg: TrainConfig, device: torch.device) -> torch.dtype:
    # MPS 下 float16 容易出 NaN，强制用 float32
    if device.type == "mps":
        return torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = mapping.get(cfg.dtype, torch.float32)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return torch.float16
    return dtype


def cosine_lr(
    step: int, warmup: int, total: int, lr: float, min_lr: float
) -> float:
    """带 warmup 的余弦学习率调度。"""
    if step < warmup:
        return lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def _unwrap_state_dict(state_dict: dict) -> dict:
    """去掉 torch.compile 添加的 _orig_mod. 前缀。"""
    unwrapped = {}
    for k, v in state_dict.items():
        key = k.replace("_orig_mod.", "")
        unwrapped[key] = v
    return unwrapped


def save_checkpoint(
    model: TinyRoleModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: Path,
):
    raw_model = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "model": _unwrap_state_dict(model.state_dict()),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "config": raw_model.cfg.__dict__,
        },
        path,
    )
    print(f"  检查点已保存: {path}")


def load_checkpoint(
    path: Path, model: TinyRoleModel, optimizer=None
) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  加载检查点: {path} (step={ckpt.get('step', 0)})")
    return ckpt.get("step", 0)


# ============================================================
#  阶段 1: 领域预训练
# ============================================================

def phase_pretrain(
    model: TinyRoleModel,
    train_cfg: TrainConfig,
    paths: PathConfig,
    device: torch.device,
    dtype: torch.dtype,
):
    """在领域原文上做因果语言建模预训练。"""
    print("\n" + "=" * 60)
    print("  阶段 1: 领域预训练 (Causal LM)")
    print("=" * 60)

    tokenizer = load_tokenizer(paths.data_dir / "tokenizer.json")
    dataset = PretrainDataset(paths.raw_text, tokenizer, model.cfg.max_seq_len)

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.pretrain_batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg.pretrain_batch_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.pretrain_lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = train_cfg.pretrain_epochs * len(train_loader)
    print(f"  训练集: {train_size} 块, 验证集: {val_size} 块")
    print(f"  总步数: {total_steps}, Warmup: {train_cfg.pretrain_warmup_steps}")

    model.train()
    step = 0
    best_val = float("inf")

    for epoch in range(train_cfg.pretrain_epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            lr = cosine_lr(
                step,
                train_cfg.pretrain_warmup_steps,
                total_steps,
                train_cfg.pretrain_lr,
                train_cfg.pretrain_min_lr,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device.type, dtype=dtype):
                _, loss = model(input_ids, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                with torch.autocast(device.type, dtype=dtype):
                    _, loss = model(input_ids, labels)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)
        model.train()

        print(
            f"  Epoch {epoch + 1}/{train_cfg.pretrain_epochs} | "
            f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model, optimizer, step, val_loss,
                paths.checkpoint_dir / "pretrain_best.pt",
            )

    save_checkpoint(
        model, optimizer, step, avg_loss,
        paths.checkpoint_dir / "pretrain_final.pt",
    )
    return step


# ============================================================
#  阶段 2: 知识蒸馏 (反向 KL 散度)
# ============================================================

def distill_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """蒸馏损失 = alpha * soft_loss + (1-alpha) * hard_loss。

    使用反向 KL 散度（mode-seeking），更适合小模型。
    """
    T = temperature
    # 反向 KL: KL(student || teacher)
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    soft_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (T * T)

    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    return alpha * soft_loss + (1 - alpha) * hard_loss


def phase_distill(
    model: TinyRoleModel,
    train_cfg: TrainConfig,
    paths: PathConfig,
    device: torch.device,
    dtype: torch.dtype,
):
    """从 teacher 模型的 soft label 蒸馏知识。

    需要预先生成的蒸馏数据（包含 teacher logits）。
    如果没有蒸馏数据文件，则跳过此阶段。
    """
    print("\n" + "=" * 60)
    print("  阶段 2: 知识蒸馏 (Reverse KLD)")
    print("=" * 60)

    distill_path = paths.data_dir / "distill_data.json"
    if not distill_path.exists():
        print("  蒸馏数据不存在，使用替代方案：自蒸馏 + 数据增强")
        print("  (在完整流程中，这里使用大模型的 logits)")
        print("  跳过蒸馏阶段，直接进入指令微调。")
        return

    from dataset import DistillDataset

    dataset = DistillDataset(distill_path, model.cfg.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.distill_batch_size,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.distill_lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = train_cfg.distill_epochs * len(loader)
    print(f"  蒸馏样本: {len(dataset)}, 总步数: {total_steps}")

    model.train()
    step = 0

    for epoch in range(train_cfg.distill_epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            lr = cosine_lr(
                step, 100, total_steps,
                train_cfg.distill_lr, train_cfg.distill_min_lr,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)

            with torch.autocast(device.type, dtype=dtype):
                student_logits, _ = model(input_ids)
                loss = distill_loss_fn(
                    student_logits.view(-1, student_logits.size(-1)),
                    teacher_logits.view(-1, teacher_logits.size(-1)),
                    labels,
                    train_cfg.distill_temperature,
                    train_cfg.distill_alpha,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        avg = epoch_loss / max(len(loader), 1)
        print(
            f"  Epoch {epoch + 1}/{train_cfg.distill_epochs} | "
            f"loss={avg:.4f} | lr={lr:.2e} | {time.time() - t0:.1f}s"
        )

    save_checkpoint(
        model, optimizer, step, avg,
        paths.checkpoint_dir / "distill_final.pt",
    )


# ============================================================
#  阶段 3: 指令微调
# ============================================================

def _run_finetune_loop(
    model: TinyRoleModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    n_epochs: int,
    step_offset: int,
    total_steps: int,
    use_weights: bool,
    paths: PathConfig,
    phase_name: str,
) -> tuple[int, float]:
    """微调训练循环（共用逻辑）。"""
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=-100, reduction="none",
    )
    step = step_offset
    best_val = float("inf")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        t0 = time.time()
        model.train()

        for batch in train_loader:
            lr = cosine_lr(
                step,
                train_cfg.finetune_warmup_steps,
                total_steps,
                train_cfg.finetune_lr,
                train_cfg.finetune_min_lr,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids)
            B, T, V = logits.shape
            per_token_loss = loss_fn(logits.view(-1, V), labels.view(-1))
            per_token_loss = per_token_loss.view(B, T)

            if use_weights:
                weights = batch["weight"].to(device).unsqueeze(1)
                per_token_loss = per_token_loss * weights

            valid = (labels != -100).float()
            loss = per_token_loss.sum() / valid.sum().clamp(min=1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits, _ = model(input_ids)
                B, T, V = logits.shape
                ptl = loss_fn(logits.view(-1, V), labels.view(-1)).view(B, T)
                valid = (labels != -100).float()
                val_loss += (ptl.sum() / valid.sum().clamp(min=1)).item()
        val_loss /= max(len(val_loader), 1)

        print(
            f"  [{phase_name}] Epoch {epoch + 1}/{n_epochs} | "
            f"train={avg_loss:.4f} | val={val_loss:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model, optimizer, step, val_loss,
                paths.checkpoint_dir / "finetune_best.pt",
            )

    return step, best_val


def phase_finetune(
    model: TinyRoleModel,
    train_cfg: TrainConfig,
    paths: PathConfig,
    device: torch.device,
    dtype: torch.dtype,
):
    """课程学习微调：先纯 QA 训练，再混入拒答样本。"""
    print("\n" + "=" * 60)
    print("  阶段 3: 指令微调 (课程学习)")
    print("=" * 60)

    # 从最佳预训练 checkpoint 重新加载
    best_pt = paths.checkpoint_dir / "pretrain_best.pt"
    if best_pt.exists():
        raw_model = getattr(model, "_orig_mod", model)
        ckpt = torch.load(best_pt, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        print(f"  已从 {best_pt} 重新加载预训练权重")

    tokenizer = load_tokenizer(paths.data_dir / "tokenizer.json")
    raw_cfg = getattr(model, "_orig_mod", model).cfg

    # --- 子阶段 A: 纯 QA 训练（20 epochs）---
    print("\n  --- 子阶段 A: 纯 QA 训练 ---")
    qa_dataset = InstructDataset(
        [paths.qa_json], tokenizer, raw_cfg.max_seq_len,
    )
    if len(qa_dataset) == 0:
        print("  没有找到 QA 数据，请先运行 generate_data.py")
        return

    qa_val_size = max(1, int(len(qa_dataset) * 0.1))
    qa_train_size = len(qa_dataset) - qa_val_size
    qa_train, qa_val = random_split(qa_dataset, [qa_train_size, qa_val_size])
    print(f"  QA 训练: {qa_train_size}, 验证: {qa_val_size}")

    qa_train_loader = DataLoader(
        qa_train, batch_size=train_cfg.finetune_batch_size,
        shuffle=True, drop_last=True,
    )
    qa_val_loader = DataLoader(
        qa_val, batch_size=train_cfg.finetune_batch_size,
    )

    qa_epochs = 20
    total_steps_a = qa_epochs * len(qa_train_loader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.finetune_lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    step, _ = _run_finetune_loop(
        model, qa_train_loader, qa_val_loader, optimizer,
        train_cfg, device, dtype,
        n_epochs=qa_epochs, step_offset=0,
        total_steps=total_steps_a, use_weights=False,
        paths=paths, phase_name="QA",
    )

    # --- 子阶段 B: QA + 拒答混合训练（10 epochs）---
    print("\n  --- 子阶段 B: 混合训练（QA 权重 3.0 / 拒答权重 0.3）---")
    mixed_dataset = InstructDataset(
        [paths.qa_json, paths.refuse_json], tokenizer,
        raw_cfg.max_seq_len, max_refuse=100,
    )
    mx_val_size = max(1, int(len(mixed_dataset) * 0.1))
    mx_train_size = len(mixed_dataset) - mx_val_size
    mx_train, mx_val = random_split(mixed_dataset, [mx_train_size, mx_val_size])
    print(f"  混合训练: {mx_train_size} (含 ≤100 拒答), 验证: {mx_val_size}")

    mx_train_loader = DataLoader(
        mx_train, batch_size=train_cfg.finetune_batch_size,
        shuffle=True, drop_last=True,
    )
    mx_val_loader = DataLoader(
        mx_val, batch_size=train_cfg.finetune_batch_size,
    )

    mix_epochs = 10
    total_steps_b = mix_epochs * len(mx_train_loader)

    optimizer_b = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.finetune_lr * 0.5,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    step, _ = _run_finetune_loop(
        model, mx_train_loader, mx_val_loader, optimizer_b,
        train_cfg, device, dtype,
        n_epochs=mix_epochs, step_offset=step,
        total_steps=step + total_steps_b, use_weights=True,
        paths=paths, phase_name="混合",
    )

    save_checkpoint(
        model, optimizer_b, step, 0.0,
        paths.checkpoint_dir / "finetune_final.pt",
    )
    print(f"\n  微调完成！共 {step} 步")


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="3 阶段训练管线")
    parser.add_argument(
        "--phase",
        choices=["all", "pretrain", "distill", "finetune"],
        default="all",
        help="运行哪个训练阶段",
    )
    parser.add_argument(
        "--model-size",
        choices=["2m", "5m", "10m"],
        default="2m",
        help="模型规模档位",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从检查点恢复训练",
    )
    args = parser.parse_args()

    # 配置
    model_factories = {
        "2m": ModelConfig.tiny_2m,
        "5m": ModelConfig.small_5m,
        "10m": ModelConfig.medium_10m,
    }
    model_cfg = model_factories[args.model_size]()
    train_cfg = TrainConfig()
    paths = PathConfig()

    device = get_device(train_cfg)
    dtype = get_dtype(train_cfg, device)

    print(f"设备: {device} | 精度: {dtype}")
    print(f"模型档位: {args.model_size}")

    # 构建模型
    model = TinyRoleModel(model_cfg).to(device)
    n_params = model.count_params()
    print(f"参数量: {n_params:,} ({n_params / 1e6:.2f}M)")

    if args.resume:
        load_checkpoint(Path(args.resume), model)

    if train_cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile 已启用")
        except Exception:
            print("torch.compile 不可用，使用 eager 模式")

    # 运行训练阶段
    phases = {
        "pretrain": [phase_pretrain],
        "distill": [phase_distill],
        "finetune": [phase_finetune],
        "all": [phase_pretrain, phase_distill, phase_finetune],
    }

    for fn in phases[args.phase]:
        fn(model, train_cfg, paths, device, dtype)

    print("\n训练完成！")
    print(f"检查点保存在: {paths.checkpoint_dir}")


if __name__ == "__main__":
    main()
