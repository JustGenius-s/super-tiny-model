"""2M 参数级 Transformer 语言模型。

采用 LLaMA 风格架构：RMSNorm + RoPE + SwiGLU + GQA，
在极小参数预算下最大化表达能力。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim: int, max_len: int, theta: float) -> torch.Tensor:
    """预计算 RoPE 频率矩阵。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(
    x: torch.Tensor, rope: torch.Tensor
) -> torch.Tensor:
    """将 RoPE 应用到 Q/K 张量。"""
    # x: (batch, n_heads, seq_len, head_dim)
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )
    rope = rope[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    out = torch.view_as_real(x_complex * rope).flatten(-2)
    return out.type_as(x)


class SwiGLU(nn.Module):
    """SwiGLU 激活的前馈网络。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        )


class Attention(nn.Module):
    """分组查询注意力 (GQA)。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        # KV 头数 = Q 头数的一半（至少 1）
        self.n_kv_heads = max(1, cfg.n_heads // 2)
        self.kv_repeat = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            cfg.d_model, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            cfg.d_model, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            cfg.d_model, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim, cfg.d_model, bias=False
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    """单个 Transformer 层：Attention + SwiGLU。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ff_norm = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope, mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class TinyRoleModel(nn.Module):
    """微型角色语言模型。

    LLaMA 风格 decoder-only Transformer，
    针对 2M 参数级别做了优化。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # 权重共享：embedding <-> lm_head
        self.lm_head.weight = self.tok_emb.weight

        head_dim = cfg.d_model // cfg.n_heads
        self.register_buffer(
            "rope",
            precompute_rope(head_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1, 1, cfg.max_seq_len, cfg.max_seq_len)),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """前向传播。

        Args:
            idx: token indices, shape (B, T)
            targets: 目标 token indices, shape (B, T)，用于计算 loss

        Returns:
            (logits, loss) — loss 在 targets 为 None 时返回 None
        """
        x = self.dropout(self.tok_emb(idx))
        for layer in self.layers:
            x = layer(x, self.rope, self.causal_mask)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """自回归生成。"""
        generated = idx
        for _ in range(max_new_tokens):
            context = generated[:, -self.cfg.max_seq_len:]
            logits, _ = self(context)
            logits = logits[:, -1, :]

            # 重复惩罚
            for token_id in set(generated[0].tolist()):
                logits[0, token_id] /= repetition_penalty

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(
                        logits, descending=True
                    )
                    cum_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    remove = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[remove] = float("-inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # EOS 停止
            if next_token.item() == 3:  # EOS token id
                break

        return generated

    def count_params(self) -> int:
        """返回可训练参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, factory in [
        ("2M", ModelConfig.tiny_2m),
        ("5M", ModelConfig.small_5m),
        ("10M", ModelConfig.medium_10m),
    ]:
        cfg = factory()
        m = TinyRoleModel(cfg)
        n = m.count_params()
        print(f"[{name}] 实际参数量: {n:,} ({n / 1e6:.2f}M)")
