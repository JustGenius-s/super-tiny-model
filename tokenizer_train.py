"""训练中文 BPE 分词器。

针对垂类语料（如《西游记》）训练小词表 BPE 分词器，
保持词表紧凑以适配 2M 参数预算。
"""

import argparse
import json
from pathlib import Path

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

from config import PathConfig

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# ChatML 格式 token
CHAT_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


def train_tokenizer(
    corpus_path: Path,
    output_dir: Path,
    vocab_size: int = 8192,
) -> Tokenizer:
    """从语料文件训练 BPE 分词器。

    Args:
        corpus_path: 纯文本语料文件路径
        output_dir: 输出目录
        vocab_size: 词表大小

    Returns:
        训练好的 Tokenizer 实例
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Replace(r"\s+", " "),
        normalizers.Strip(),
    ])

    # 按字符切分（适合中文），空格单独处理
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.UnicodeScripts(),
    ])

    all_special = SPECIAL_TOKENS + CHAT_TOKENS
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=all_special,
        min_frequency=2,
        show_progress=True,
    )

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"语料文件 {corpus_path} 不存在。\n"
            "请先将原始文本放入 data/corpus.txt"
        )

    tokenizer.train([str(corpus_path)], trainer)

    # 后处理：添加 BOS/EOS
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        pair=f"<bos>:0 $A:0 <eos>:0 <bos>:1 $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", bos_id),
            ("<eos>", eos_id),
        ],
    )

    tokenizer.decoder = decoders.BPEDecoder()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "tokenizer.json"
    tokenizer.save(str(save_path))

    # 保存词表摘要
    vocab = tokenizer.get_vocab()
    summary = {
        "vocab_size": len(vocab),
        "special_tokens": {t: tokenizer.token_to_id(t) for t in all_special},
        "sample_tokens": dict(list(sorted(vocab.items(), key=lambda x: x[1]))[:50]),
    }
    with open(output_dir / "tokenizer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"分词器已保存到 {save_path}")
    print(f"词表大小: {len(vocab)}")

    # 测试编码
    test_texts = [
        '悟空道：\u201c师父，前面有妖怪。\u201d',
        '三藏闻言大惊。',
        '你好，请问今天天气如何？',
    ]
    print("\n--- 编码测试 ---")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"原文: {text}")
        print(f"Token数: {len(encoded.ids)}, IDs: {encoded.ids[:20]}")
        print(f"解码: {decoded}\n")

    return tokenizer


def load_tokenizer(path: Path | None = None) -> Tokenizer:
    """加载已训练的分词器。"""
    if path is None:
        path = PathConfig().data_dir / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(
            f"分词器 {path} 不存在，请先运行 tokenizer_train.py"
        )
    return Tokenizer.from_file(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 BPE 分词器")
    parser.add_argument(
        "--corpus", type=str, default=None,
        help="语料文件路径 (默认: data/corpus.txt)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=8192,
        help="词表大小 (默认: 8192)",
    )
    args = parser.parse_args()

    paths = PathConfig()
    corpus = Path(args.corpus) if args.corpus else paths.raw_text
    train_tokenizer(corpus, paths.data_dir, args.vocab_size)
