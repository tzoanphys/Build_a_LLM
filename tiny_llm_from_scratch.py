import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 1) Tiny character tokenizer
# ----------------------------
class CharTokenizer:
    """
    Character-level tokenizer: each character is a token.
    Builds vocab from provided training texts.
    """

    def __init__(self, texts: List[str]):
        chars = sorted(list(set("".join(texts))))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


# ----------------------------
# 2) Training data formatting
# ----------------------------
def make_training_samples(pairs: List[Tuple[str, str]]) -> List[str]:
    """
    Convert (word, sentence) pairs into training strings.
    We add a delimiter ### so generation knows when to stop.
    """
    samples = []
    for w, sent in pairs:
        samples.append(f"Word: {w}\nSentence: {sent}\n###\n")
    return samples


# ----------------------------
# 3) Mini GPT-like Transformer
# ----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.block_size = block_size

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

        # Causal mask (precomputed): shape (1, 1, T, T)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        """
        x: (B, T, C)
        """
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # reshape for multi-head: (B, nh, T, hd)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # attention scores: (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal masking so token t cannot see future tokens > t
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual
        x = x + self.ff(self.ln2(x))    # residual
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layers: int, n_heads: int, n_embd: int, dropout: float):
        super().__init__()
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) int tokens
        targets: (B, T) next-token labels
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        tok = self.token_emb(idx)  # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, C)
        x = self.drop(tok + pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0):
        """
        Autoregressive generation.
        idx: (1, T) prompt tokens
        """
        for _ in range(max_new_tokens):
            # crop to last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (1, vocab)

            if top_k > 0:
                k = min(top_k, logits.size(-1))  # clamp to vocab size
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx


# ----------------------------
# 4) Batching utilities
# ----------------------------
def build_stream(samples: List[str]) -> str:
    # Join everything into one long training stream
    return "".join(samples)

def get_batch(data_ids: torch.Tensor, block_size: int, batch_size: int, device: str):
    """
    Pick random starting points in the big 1D token stream.
    Return x (context) and y (next-token targets).
    """
    max_start = data_ids.size(0) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data_ids[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data_ids[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y


# ----------------------------
# 5) Training + demo
# ----------------------------
@dataclass
class TrainConfig:
    block_size: int = 256
    batch_size: int = 32
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    lr: float = 3e-4
    steps: int = 2000
    eval_every: int = 200
    seed: int = 42


def main():
    torch.manual_seed(TrainConfig.seed)
    random.seed(TrainConfig.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # --- Tiny dataset (you can expand this a lot) ---
    pairs = [
        ("apple", "There is a legacy that a falling apple inspired Newton."),
        ("gravity", "Gravity is the force that pulls objects toward each other."),
        ("telescope", "A telescope helps us see distant objects in the night sky."),
        ("ocean", "The ocean covers most of Earth and shapes climate and weather."),
        ("electricity", "Electricity powers devices by moving charge through circuits."),
        ("moon", "The Moon orbits Earth and affects tides through gravity."),
        ("rain", "Rain forms when water vapor condenses into droplets and falls."),
        ("volcano", "A volcano erupts when pressure pushes magma to the surface."),
    ]

    samples = make_training_samples(pairs)
    text_stream = build_stream(samples)

    tokenizer = CharTokenizer([text_stream])
    data_ids = torch.tensor(tokenizer.encode(text_stream), dtype=torch.long)

    cfg = TrainConfig()
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    def estimate_loss(iters=50):
        model.eval()
        losses = []
        for _ in range(iters):
            xb, yb = get_batch(data_ids, cfg.block_size, cfg.batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    print("Vocab size:", tokenizer.vocab_size)
    print("Training tokens:", data_ids.numel())

    for step in range(1, cfg.steps + 1):
        xb, yb = get_batch(data_ids, cfg.block_size, cfg.batch_size, device)
        _, loss = model(xb, yb)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % cfg.eval_every == 0:
            train_loss = estimate_loss(iters=30)
            print(f"step {step}/{cfg.steps} | loss ~ {train_loss:.4f}")

    # --- Demo generation ---
    def run_query(word: str):
        prompt = f"Word: {word}\nSentence: "
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

        out = model.generate(idx, max_new_tokens=200, temperature=0.9, top_k=0) #out = model.generate(idx, max_new_tokens=200, temperature=0.9, top_k=50)
        decoded = tokenizer.decode(out[0].tolist())

        # Stop at delimiter if present
        stop = "\n###"
        if stop in decoded:
            decoded = decoded.split(stop)[0]

        print("\n--- QUERY ---")
        print(word)
        print("--- OUTPUT ---")
        print(decoded)

    run_query("apple")
    run_query("moon")
    run_query("electricity")


if __name__ == "__main__":
    main()