import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # q, k, v juntos
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # máscara causal: cada token só enxerga os anteriores
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                       # soma ponderada dos valores
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concatena as cabeças
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")  # o GPT-2 usa a versão tanh
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # pre-norm: LayerNorm antes, fluxo residual limpo
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024   # tamanho máximo da sequência
    vocab_size: int = 50257  # 50000 merges BPE + 256 bytes + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # embeddings de tokens
            wpe=nn.Embedding(config.block_size, config.n_embd),   # embeddings de posição
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),                     # LayerNorm final
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ADICIONADO: Weight Sharing (Compartilhamento de pesos)
        # No GPT-2, os pesos do embedding de entrada e da projeção linear de saída são os mesmos
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        # idx tem formato (B, T): B sequências de T índices de tokens
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequência de tamanho {T} maior que o block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # posições 0..T-1
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        x = tok_emb + pos_emb                 # o pos_emb é replicado para todo o batch
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)              # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Carrega os pesos do GPT-2 da HuggingFace na nossa classe."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),    # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),   # 350M
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        model = GPT(GPTConfig(**config_args))
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]  # ignora a máscara

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")]

        # a OpenAI usava Conv1D (TensorFlow); o nn.Linear do PyTorch pede o peso transposto
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight",
                      "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(sd_keys_hf) == len(sd_keys), f"chaves diferentes: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


# -----------------------------------------------------------------------------
# ADICIONADO: Script de teste e geração de texto ao executar o arquivo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Instancia o modelo carregando os pesos oficiais do GPT-2 (124M)
    model = GPT.from_pretrained("gpt2")
    model.eval()
    model.to(device)

    # Tokenização do prompt de entrada
    enc = tiktoken.get_encoding("gpt2")
    prompt = "Hello, I'm a language model,"
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Gera texto até atingir 30 tokens
    torch.manual_seed(42)
    while x.size(1) < 30:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]  # Seleciona os logits do último token
            probs = F.softmax(logits, dim=-1)
            
            # Amostragem top-k (k=50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            
            # Anexa o token sorteado à sequência de entrada
            x = torch.cat((x, xcol), dim=1)

    # Imprime a saída final decodificada
    print("\n--- Texto Gerado ---")
    print(enc.decode(x[0].tolist()))