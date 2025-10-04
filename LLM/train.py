import math, os, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .MinGPT import MiniGPT

class CFG:
    vocab_size        = 10000
    context_length    = 128
    d_model           = 256
    n_heads           = 4
    n_blocks          = 4
    dropout           = 0.1

    batch_size        = 16
    epochs            = 1
    lr                = 3e-4
    betas             = (0.9, 0.95)
    weight_decay      = 0.1
    grad_clip         = 1.0
    log_every         = 200
    device            = "mps" if torch.backends.mps.is_available() else "cpu"
    limit_train_chars = 300_000   
    limit_val_chars   = 30_000

cfg = CFG()

class TokenBlockDataset(Dataset):
    """
    Given one long list of token IDs, returns pairs (x, y) of length T:
      x = ids[i : i+T]
      y = ids[i+1 : i+T+1]
    So y is x shifted left by 1.
    """
    def __init__(self, ids, block_size):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.T = block_size
        self.n = len(self.ids) - self.T  # last start idx is len - (T+1)

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, i):
        x = self.ids[i : i + self.T]               # (T,)
        y = self.ids[i + 1 : i + 1 + self.T]       # (T,)
        return x, y

def make_loaders(token_ids, val_frac=0.1):
    split = int(len(token_ids) * (1 - val_frac))
    train_ids, val_ids = token_ids[:split], token_ids[split:]
    train_ds = TokenBlockDataset(train_ids, cfg.context_length)
    val_ds   = TokenBlockDataset(val_ids,   cfg.context_length)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    return train_dl, val_dl

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x = x.to(cfg.device)                 # (B, T)
        y = y.to(cfg.device)                 # (B, T)
        logits = model(x)                    # (B, T, V)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B*T, V), y.view(B*T))
        total += loss.item()
        count += 1
    avg = total / max(1, count)
    ppl = math.exp(avg)
    return avg, ppl

def train(model, train_dl, val_dl):
    model.to(cfg.device)
    loss_fn  = nn.CrossEntropyLoss()
    optim    = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                 betas=cfg.betas, weight_decay=cfg.weight_decay)

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_dl:
            x = x.to(cfg.device)                     # (B, T)
            y = y.to(cfg.device)                     # (B, T)
            logits = model(x)                        # (B, T, V)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B*T, V), y.view(B*T))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            if step % cfg.log_every == 0:
                print(f"step {step:6d} | train_loss {loss.item():.4f}")
            step += 1

        val_loss, val_ppl = evaluate(model, val_dl, loss_fn)
        print(f"[epoch {epoch+1}] val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": vars(cfg)}, "checkpoints/minigpt.pt")
    print("Saved -> checkpoints/minigpt.pt")

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=None):
    model.eval()
    x = torch.tensor(prompt_ids, dtype=torch.long, device=cfg.device)[None, :]  # (1, t)
    for _ in range(max_new_tokens):
        x_cond = x[:, -cfg.context_length:]                # crop to block size
        logits = model(x_cond)                             # (1, t', V)
        next_logits = logits[:, -1, :] / max(1e-8, temperature)
        if top_k is not None:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -1e10
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        x = torch.cat([x, next_id], dim=1)
    return x[0].tolist()

if __name__ == "__main__":
    from ..Tokenizer.RegexTokenizer import Tokenizer
    from datasets import load_dataset
    import io, os

    tok = Tokenizer()
    tok.load("MiniGPT/Tokenizer/models/learned_tokenizer.json")

    cfg.vocab_size = tok._get_vocab_size()
    
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    train_text = "\n".join(ds["train"]["text"])[:CFG.limit_train_chars]
    val_text   = "\n".join(ds["validation"]["text"])[:CFG.limit_val_chars]
    full_text  = train_text + "\n" + val_text

    token_ids = tok.encode(full_text)[: CFG.context_length * CFG.batch_size * 2000]

    train_dl, val_dl = make_loaders(token_ids)
    model = MiniGPT(vocab_size=cfg.vocab_size,
                    context_length=cfg.context_length,
                    d_model=cfg.d_model,
                    n_blocks=cfg.n_blocks,
                    n_heads=cfg.n_heads,
                    dropout=cfg.dropout)
    train(model, train_dl, val_dl)
    
    # Sampling:
    prompt = "Write me a poem."
    prompt_ids = tok.encode(prompt)
    out_ids = generate(model, prompt_ids, max_new_tokens=50, temperature=0.8, top_k=50)
    print(tok.decode(out_ids))