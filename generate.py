"""
generate.py  —  Interactive text generation from a saved GPT checkpoint.

Usage:
    python generate.py
    python generate.py --checkpoint gpt_swahili.pt --tokens 300 --temperature 0.8
    python generate.py --prompt "Rais wa Tanzania" --tokens 500
"""

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

#  CLI args 
parser = argparse.ArgumentParser(description="Generate Swahili text from a trained GPT checkpoint.")
parser.add_argument("--checkpoint",   default="gpt_swahili.pt", help="Path to .pt checkpoint file")
parser.add_argument("--tokens",       type=int,   default=500,  help="Number of characters to generate")
parser.add_argument("--temperature",  type=float, default=1.0,  help="Sampling temperature (0.5–1.5 recommended)")
parser.add_argument("--prompt",       type=str,   default="",   help="Seed text (leave empty for unconditional)")
args = parser.parse_args()

device = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)


#  Load checkpoint 
print(f"Loading checkpoint: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=device)

cfg        = ckpt["config"]
stoi       = ckpt["vocab"]["stoi"]
itos       = ckpt["vocab"]["itos"]
encode     = lambda s: [stoi.get(c, 0) for c in s]
decode     = lambda l: "".join([itos[i] for i in l])

n_embd     = cfg["n_embd"]
n_head     = cfg["n_head"]
n_layer    = cfg["n_layer"]
block_size = cfg["block_size"]
vocab_size = cfg["vocab_size"]
dropout    = 0.0   # disabled at inference


#  Rebuild model 
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key     = nn.Linear(n_embd, head_size, bias=False)
        self.query   = nn.Linear(n_embd, head_size, bias=False)
        self.value   = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return self.dropout(wei) @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa   = MultiHeadAttention(n_head, n_embd // n_head)
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
        x       = self.token_embedding_table(idx) + \
                  self.position_embedding_table(torch.arange(T, device=device))
        logits  = self.lm_head(self.ln_final(self.blocks(x)))
        loss    = None
        if targets is not None:
            B, T, C = logits.shape
            loss    = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits     = logits[:, -1, :] / temperature
            probs      = F.softmax(logits, dim=-1)
            idx        = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


model = GPTLanguageModel().to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model loaded  ({sum(p.numel() for p in model.parameters()):,} parameters)\n")

#  Generate 
def generate(prompt: str = "", n_tokens: int = 500, temperature: float = 1.0) -> str:
    if prompt:
        tokens = encode(prompt)
        ctx    = torch.tensor([tokens], dtype=torch.long, device=device)
    else:
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)

    out = model.generate(ctx, max_new_tokens=n_tokens, temperature=temperature)
    return decode(out[0].tolist())


#  Interactive loop 
if args.prompt:
    # single-shot mode
    print(generate(args.prompt, args.tokens, args.temperature))
else:
    print("Interactive mode — type a Swahili prompt and press Enter.")
    print("Leave prompt empty for unconditional generation.  Ctrl-C to quit.\n")
    while True:
        try:
            prompt = input("Prompt > ").strip()
            print("\n" + generate(prompt, args.tokens, args.temperature) + "\n")
        except KeyboardInterrupt:
            print("\nKwaheri!")
            break
