"""交互式对话脚本。

用法:
    python chat.py                          # 使用默认检查点
    python chat.py --checkpoint path/to.pt  # 指定检查点
    python chat.py --model-size 10m         # 指定模型规模
"""

import argparse
from pathlib import Path

import torch

from config import ModelConfig, PathConfig
from model import TinyRoleModel
from tokenizer_train import load_tokenizer


def find_best_checkpoint(paths: PathConfig) -> Path | None:
    """按优先级查找最佳检查点。"""
    candidates = [
        "finetune_best.pt",
        "finetune_final.pt",
        "distill_final.pt",
        "pretrain_best.pt",
        "pretrain_final.pt",
    ]
    for name in candidates:
        p = paths.checkpoint_dir / name
        if p.exists():
            return p
    return None


def format_prompt(
    question: str,
    system_prompt: str,
) -> str:
    """构造 ChatML 格式输入。"""
    return (
        f"<|system|>{system_prompt}<|end|>"
        f"<|user|>{question}<|end|>"
        f"<|assistant|>"
    )


def main():
    parser = argparse.ArgumentParser(description="交互式对话")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--model-size", choices=["2m", "5m", "10m"], default="2m",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "你是一个专业的角色助手，只回答与你的知识领域相关的问题。"
            "对于领域外的问题，请礼貌地拒绝。"
        ),
    )
    args = parser.parse_args()

    paths = PathConfig()

    # 加载分词器
    tokenizer = load_tokenizer(paths.data_dir / "tokenizer.json")

    # 加载模型
    factories = {
        "2m": ModelConfig.tiny_2m,
        "5m": ModelConfig.small_5m,
        "10m": ModelConfig.medium_10m,
    }

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(paths)
    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "config" in ckpt:
            model_cfg = ModelConfig(**ckpt["config"])
        else:
            model_cfg = factories[args.model_size]()
        model = TinyRoleModel(model_cfg)
        model.load_state_dict(ckpt["model"])
        print(f"已加载检查点: {ckpt_path}")
    else:
        model_cfg = factories[args.model_size]()
        model = TinyRoleModel(model_cfg)
        print("未找到检查点，使用随机初始化模型（仅用于测试）")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device).eval()
    n_params = model.count_params()
    print(f"参数量: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"设备: {device}")
    print(f"输入 'quit' 或 'exit' 退出\n")

    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        prompt = format_prompt(question, args.system_prompt)
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

        # 只解码新生成的 token
        new_ids = output_ids[0, input_ids.shape[1]:].tolist()
        # 截断到 <|end|> 或 <eos>
        end_token = tokenizer.token_to_id("<|end|>")
        eos_token = tokenizer.token_to_id("<eos>")
        for i, tid in enumerate(new_ids):
            if tid in (end_token, eos_token):
                new_ids = new_ids[:i]
                break

        response = tokenizer.decode(new_ids)
        print(f"助手: {response}\n")


if __name__ == "__main__":
    main()
