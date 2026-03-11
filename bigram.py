"""
bigram.py  —  Character-level Bigram Language Model
Trained on the Swahili News corpus.

Architecture
------------
- Token embedding table  (vocab_size × n_embed)
- Positional embedding   (block_size  × n_embed)
- Linear language head   (n_embed     × vocab_size)

The model is a simple baseline: each token attends only to itself and
predicts the next character via a learned lookup table. No attention.

Usage:
    python data/prepare.py          # download corpus first
    python bigram.py                # train and generate
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# ── Hyperparameters ────────────────────────────────────────────────────────────
batch_size    = 32      # independent sequences processed in parallel
block_size    = 8       # maximum context length
max_iters     = 10_000
learning_rate = 1e-3
eval_interval = 500
eval_iters    = 200
n_embed       = 32

device = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)
torch.manual_seed(1337)
# ──────────────────────────────────────────────────────────────────────────────

with open("data/swahili.txt", "r", encoding="utf-8") as f:
    text = f.read()

# character-level vocabulary
chars      = sorted(set(text))
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}
itos       = {i: ch for i, ch in enumerate(chars)}
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda l: "".join([itos[i] for i in l])

# train / val split  (80 / 20)
data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.8 * len(data))
train_data = data[:n]
val_data   = data[n:]


def get_batch(split):
    src  = train_data if split == "train" else val_data
    ix   = torch.randint(len(src) - block_size, (batch_size,))
    x    = torch.stack([src[i:i + block_size]     for i in ix])
    y    = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y       = get_batch(split)
            _, loss    = model(X, Y)
            losses[k]  = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head                  = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T    = idx.shape
        tok_emb = self.token_embedding_table(idx)                                # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x       = tok_emb + pos_emb                                              # (B, T, C)
        logits  = self.lm_head(x)                                                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss    = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond   = idx[:, -block_size:]                # crop to block_size
            logits, _  = self(idx_cond)
            logits      = logits[:, -1, :]                   # (B, C)
            probs       = F.softmax(logits, dim=-1)          # (B, C)
            idx_next    = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx         = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# ── Training ──────────────────────────────────────────────────────────────────
model     = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses, val_losses, steps_log = [], [], []

print(f"Training Bigram model on {device}  |  vocab_size={vocab_size}")
print(f"Dataset: {len(text):,} chars  |  {len(train_data):,} train  {len(val_data):,} val\n")

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {step:5d}: train {losses['train']:.4f}  val {losses['val']:.4f}")
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        steps_log.append(step)

    xb, yb    = get_batch("train")
    _, loss   = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nFinal loss: {loss.item():.4f}")

# ── Plot training curves ───────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(steps_log, train_losses, label="train")
plt.plot(steps_log, val_losses,   label="val")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Bigram — Swahili News Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig("bigram_loss.png", dpi=150)
print("Saved bigram_loss.png")

# ── Generate sample ───────────────────────────────────────────────────────────
print("\n── Generated text (500 chars) ─────────────────────────────────────────")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
