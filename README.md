# Super Tiny Model

用 2M 参数训练一个完全垂类的角色 AI —— 通过知识蒸馏从大模型中提取单一领域知识。

## 核心理念

角色扮演不需要百亿参数的通用大模型。一个 2M 参数的微型 Transformer，只要训练方法得当，就能精通单一知识域（如《西游记》原文），并对域外问题坚决拒答。

## 架构

- **LLaMA 风格** decoder-only Transformer
- RMSNorm + RoPE 旋转位置编码 + SwiGLU 激活 + GQA 分组查询注意力
- 权重共享（embedding ↔ lm_head）

| 档位 | 参数量 | d_model | n_layers | d_ff | 适用场景 |
|------|--------|---------|----------|------|---------|
| 2M   | ~2M    | 128     | 6        | 512  | 极窄垂类（单本书） |
| 5M   | ~5M    | 192     | 8        | 768  | 中等垂类（一个主题） |
| 10M  | ~10M   | 256     | 10       | 1024 | 较宽垂类（多角色） |

## 3 阶段训练管线

```
阶段 1: 领域预训练 ─── 在原文上做因果语言建模，学习语言模式
    │
阶段 2: 知识蒸馏 ──── 从大模型 (GPT-4o) 的 soft label 中学习，反向 KL 散度
    │
阶段 3: 指令微调 ──── QA 对 + 拒答样本，学会精准回答和拒绝
```

## 快速开始

### 1. 安装依赖

```bash
cd super-tiny-model
pip install -r requirements.txt
```

### 2. 准备语料

将原始文本放入 `data/corpus.txt`（例如《西游记》全文）。

### 3. 训练分词器

```bash
python tokenizer_train.py --vocab-size 8192
```

### 4. 生成训练数据

```bash
export OPENAI_API_KEY="your-key"
python generate_data.py \
    --role-name "西游记原文助手" \
    --domain "《西游记》原著内容" \
    --model gpt-4o-mini
```

支持通过 `--base-url` 使用任何 OpenAI 兼容接口。

### 5. 训练模型

```bash
# 完整 3 阶段
python train.py --phase all --model-size 2m

# 或分阶段运行
python train.py --phase pretrain --model-size 2m
python train.py --phase finetune --model-size 2m
```

### 6. 对话测试

```bash
python chat.py --model-size 2m
```

## 项目结构

```
super-tiny-model/
├── config.py           # 模型 & 训练超参数
├── model.py            # 2M 参数 Transformer 实现
├── tokenizer_train.py  # BPE 分词器训练
├── dataset.py          # 数据集类（预训练/指令/蒸馏）
├── generate_data.py    # 用大模型 API 生成训练数据
├── train.py            # 3 阶段训练管线
├── chat.py             # 交互式对话
├── requirements.txt    # Python 依赖
└── data/
    ├── corpus.txt      # 原始语料（需自行放入）
    ├── tokenizer.json  # 训练后的分词器
    ├── qa_pairs.json   # QA 训练数据
    └── refuse_pairs.json # 拒答训练数据
```

## 替换为其他角色

只需修改三样东西：

1. `data/corpus.txt` → 换成目标领域的原始文本
2. `generate_data.py` 的 `--role-name` 和 `--domain` 参数
3. `chat.py` 的 `--system-prompt` 参数

示例：

```bash
# 训练一个"伤寒论"医学助手
python generate_data.py \
    --role-name "伤寒论助手" \
    --domain "《伤寒论》原文内容"
```

## 硬件需求

| 档位 | 训练 | 推理 |
|------|------|------|
| 2M   | 任意 CPU / 入门级 GPU | CPU 即可 |
| 5M   | 4GB+ VRAM GPU | CPU 即可 |
| 10M  | 8GB+ VRAM GPU | CPU / 手机 |
