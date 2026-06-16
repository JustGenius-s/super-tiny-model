"""使用大模型 API 从原始文本生成蒸馏/微调数据。

支持两种模式：
1. QA 生成：从原文段落中提取问答对
2. 拒答生成：生成领域外问题及标准拒答回复
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

from config import PathConfig

QA_SYSTEM_PROMPT = """你是一个数据标注专家。给你一段古典文学原文，请生成高质量的问答对。

要求：
1. 问题必须可以从给定原文中找到明确答案
2. 答案要精准引用或概括原文内容，不要编造
3. 问题类型要多样：人物、情节、对话、地点、细节等
4. 每个段落生成 3-5 个问答对

请严格按如下 JSON 数组格式输出，不要输出任何其他内容：
[{"question": "...", "answer": "..."}]
"""

REFUSE_SYSTEM_PROMPT = """你是一个数据标注专家。请生成与"目标领域"无关的日常问题，
以及对应的标准拒答回复。

拒答模板：
"这个问题不在我的知识范围内。我是{role_name}，只能回答关于{domain}的问题。"

要求：
1. 问题要多样化：天气、数学、编程、时事、科学、日常生活等
2. 每次生成 10 个问答对

请严格按如下 JSON 数组格式输出，不要输出任何其他内容：
[{"question": "...", "answer": "..."}]
"""


def split_into_chunks(text: str, chunk_size: int = 800) -> list[str]:
    """将原文按段落切分为合适大小的块。"""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) > chunk_size and current:
            chunks.append(current)
            current = p
        else:
            current = current + "\n" + p if current else p
    if current:
        chunks.append(current)
    return chunks


def _extract_json(text: str) -> list[dict]:
    """从模型回复中提取 JSON 数组，兼容 markdown 代码块包裹。"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("pairs", "qa_pairs", "data", "questions"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return []
    return data if isinstance(data, list) else []


def generate_qa_pairs(
    client: OpenAI,
    chunk: str,
    model: str = "deepseek-chat",
) -> list[dict]:
    """从一段原文生成 QA 对。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": f"原文段落：\n\n{chunk}"},
            ],
            temperature=0.7,
        )
        content = resp.choices[0].message.content
        return _extract_json(content)
    except Exception as e:
        print(f"  生成失败: {e}")
        return []


def generate_refuse_pairs(
    client: OpenAI,
    role_name: str,
    domain: str,
    count: int = 10,
    model: str = "deepseek-chat",
) -> list[dict]:
    """生成拒答样本。"""
    prompt = REFUSE_SYSTEM_PROMPT.replace("{role_name}", role_name)
    prompt = prompt.replace("{domain}", domain)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"请生成 {count} 个领域外的问题和拒答回复。",
                },
            ],
            temperature=0.9,
        )
        content = resp.choices[0].message.content
        return _extract_json(content)
    except Exception as e:
        print(f"  生成失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="生成训练数据")
    parser.add_argument(
        "--corpus", type=str, default=None,
        help="原始语料文件路径",
    )
    parser.add_argument(
        "--role-name", type=str, default="西游记原文助手",
        help="角色名称",
    )
    parser.add_argument(
        "--domain", type=str, default="《西游记》原著内容",
        help="知识领域描述",
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-chat",
        help="调用的 LLM 模型名",
    )
    parser.add_argument(
        "--refuse-batches", type=int, default=30,
        help="拒答数据生成批次（每批 10 条）",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenAI API Key (也可用环境变量 OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="API base URL（兼容其他 OpenAI 兼容接口）",
    )
    args = parser.parse_args()

    paths = PathConfig()
    corpus_path = Path(args.corpus) if args.corpus else paths.raw_text

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：请设置 OPENAI_API_KEY 环境变量或使用 --api-key 参数")
        return

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    # --- 生成 QA 对 ---
    if corpus_path.exists():
        print(f"从 {corpus_path} 生成 QA 数据...")
        text = corpus_path.read_text(encoding="utf-8")
        chunks = split_into_chunks(text)
        print(f"  切分为 {len(chunks)} 个文本块")

        all_qa: list[dict] = []
        for i, chunk in enumerate(chunks):
            print(f"  处理第 {i + 1}/{len(chunks)} 块...", end=" ")
            pairs = generate_qa_pairs(client, chunk, args.model)
            all_qa.extend(pairs)
            print(f"生成 {len(pairs)} 个 QA 对")
            time.sleep(0.5)  # rate limit

        qa_path = paths.qa_json
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(all_qa, f, ensure_ascii=False, indent=2)
        print(f"\nQA 数据已保存: {qa_path} ({len(all_qa)} 条)")
    else:
        print(f"语料文件 {corpus_path} 不存在，跳过 QA 生成")

    # --- 生成拒答数据 ---
    print(f"\n生成拒答数据 ({args.refuse_batches} 批)...")
    all_refuse: list[dict] = []
    for i in range(args.refuse_batches):
        print(f"  第 {i + 1}/{args.refuse_batches} 批...", end=" ")
        pairs = generate_refuse_pairs(
            client, args.role_name, args.domain, 10, args.model
        )
        all_refuse.extend(pairs)
        print(f"生成 {len(pairs)} 条")
        time.sleep(0.5)

    refuse_path = paths.refuse_json
    with open(refuse_path, "w", encoding="utf-8") as f:
        json.dump(all_refuse, f, ensure_ascii=False, indent=2)
    print(f"\n拒答数据已保存: {refuse_path} ({len(all_refuse)} 条)")

    # --- 汇总 ---
    total = len(all_qa) + len(all_refuse) if corpus_path.exists() else len(all_refuse)
    qa_count = len(all_qa) if corpus_path.exists() else 0
    print(f"\n=== 数据生成完成 ===")
    print(f"QA 对: {qa_count}")
    print(f"拒答对: {len(all_refuse)}")
    print(f"总计: {total}")


if __name__ == "__main__":
    main()
