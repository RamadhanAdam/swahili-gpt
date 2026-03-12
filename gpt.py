"""
gpt.py  —  Characterlevel GPT Language Model
Trained on the Swahili News corpus.

Architecture  (decoderonly Transformer)

  Token embedding    → vocab_size × n_embd
  Position embedding → block_size × n_embd
  N × Transformer Block:
      LayerNorm → MultiHead SelfAttention (causal) → residual
      LayerNorm → FeedForward (4× expansion, ReLU)  → residual
  LayerNorm → Linear head → vocab_size

Default config trains a ~10M parameter model.
Reduce n_embd / n_layer / n_head for faster experiments.

Usage:
    python data/prepare.py          # download corpus first
    python gpt.py                   # train
    python generate.py              # interactive generation from saved checkpoint
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

#  Hyperparameters 
batch_size    = 64     # sequences per batch
block_size    = 256    # context window (characters)
max_iters     = 5_000
eval_interval = 500
eval_iters    = 200
learning_rate = 3e4

n_embd   = 384
n_head   = 6
n_layer  = 6
dropout  = 0.2

CHECKPOINT = "gpt_swahili.pt"   # saved after training

device = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)
torch.manual_seed(344)
# 

with open("data/swahili.txt", "r", encoding="utf8") as f:
    text = f.read()

# characterlevel vocabulary
chars      = sorted(set(text))
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}
itos       = {i: ch for i, ch in enumerate(chars)}
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda l: "".join([itos[i] for i in l])

# train / val split  (90 / 10)
data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]


def get_batch(split):
    src = train_data if split == "train" else val_data
    ix  = torch.randint(len(src) - block_size, (batch_size,))
    x   = torch.stack([src[i:i + block_size]         for i in ix])
    y   = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y      = get_batch(split)
            _, loss   = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


#  Model components 

class Head(nn.Module):
    """Single causal selfattention head."""

    def __init__(self, head_size):
        super().__init__()
        self.key     = nn.Linear(n_embd, head_size, bias=False)
        self.query   = nn.Linear(n_embd, head_size, bias=False)
        self.value   = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # causal mask: lowertriangular matrix stored as a nonparameter buffer
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # scaled dotproduct attention
        scale = C ** 0.5
        wei   = q @ k.transpose(2, 1) * scale             # (B, T, T)
        wei   = wei.masked_fill(self.tril[:T, :T] == 0, float("inf"))
        wei   = F.softmax(wei, dim=1)
        wei   = self.dropout(wei)

        v   = self.value(x)   # (B, T, head_size)
        out = wei @ v          # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Concatenated parallel attention heads + projection."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Positionwise feedforward network (4× expansion)."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: LayerNorm → Attention → residual → LayerNorm → FFN → residual."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks   = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head  = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T    = idx.shape
        tok_emb = self.token_embedding_table(idx)                                # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x       = self.ln_final(self.blocks(tok_emb + pos_emb))                 # (B, T, C)
        logits  = self.lm_head(x)                                                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss    = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Autoregressive generation.

        Args:
            idx:            (B, T) seed token indices
            max_new_tokens: number of characters to generate
            temperature:    >1 → more random, <1 → more focused
        """
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, block_size:]
            logits, _ = self(idx_cond)
            logits     = logits[:, 1, :] / temperature          # (B, C)
            probs      = F.softmax(logits, dim=1)
            idx_next   = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx        = torch.cat((idx, idx_next), dim=1)
        return idx


#  Training 
model     = GPTLanguageModel().to(device)
n_params  = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"GPT — {n_params:,} parameters  |  device: {device}")
print(f"Dataset: {len(text):,} chars  |  vocab_size={vocab_size}")
print(f"Config: n_embd={n_embd}  n_head={n_head}  n_layer={n_layer}  block_size={block_size}\n")

train_losses, val_losses, steps_log = [], [], []

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {step:5d}: train {losses['train']:.4f}  val {losses['val']:.4f}")
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        steps_log.append(step)

    xb, yb   = get_batch("train")
    _, loss  = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nFinal loss: {loss.item():.4f}")

#  Save checkpoint 
torch.save({
    "model_state": model.state_dict(),
    "vocab":       {"stoi": stoi, "itos": itos},
    "config": {
        "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
        "block_size": block_size, "vocab_size": vocab_size,
    },
}, CHECKPOINT)
print(f"Checkpoint saved → {CHECKPOINT}")

#  Plot training curves 
plt.figure(figsize=(8, 4))
plt.plot(steps_log, train_losses, label="train")
plt.plot(steps_log, val_losses,   label="val")
plt.xlabel("Step")
plt.ylabel("Crossentropy loss")
plt.title("GPT — Swahili News Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig("gpt_loss.png", dpi=150)
print("Saved gpt_loss.png")

#  Generate sample 
print("\n Generated text (500 chars) ")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
