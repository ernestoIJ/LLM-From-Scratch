# LLM-From-Scratch (MiniGPT)

A minimal, readable GPT-style language model built end-to-end: **custom tokenizer ➜ Baby GPT ➜ training ➜ generation**. Designed for learning and interview-ready explanations, with production-adjacent patterns and clear shapes.

---

## Highlights
- From-scratch **BPE tokenizer** with a saved JSON model.
- **Pre-Norm Transformer blocks** (LayerNorm → Attention → Residual; LayerNorm → MLP → Residual).
- Clear **single-head** and **multi-head** attention implementations.
- Clean **training loop** for next-token prediction (use `CrossEntropyLoss` on raw logits).
- Tiny **generator** with temperature / top-k sampling.
- Runs on CPU, macOS **MPS**, or CUDA if available.
