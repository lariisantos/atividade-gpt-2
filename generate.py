import torch
import torch.nn.functional as F
import tiktoken
from train_gpt2 import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"A usar dispositivo: {device}")

# Carrega os pesos pré-treinados do GPT-2 oficial na tua classe otimizada
model = GPT.from_pretrained("gpt2")
model.to(device)
model.eval()

def generate_text(prompt, max_new_tokens=40, top_k=50):
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :50257]
            probs = F.softmax(logits, dim=-1)
            
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    return enc.decode(x[0].tolist())

prompts = [
    "Hello, I'm a language model,",
    "In a shocking finding, scientist discovered a herd of unicorns",
    "The secret to building a great software project is"
]

for p in prompts:
    print(f"=== Prompt: '{p}' ===")
    print(generate_text(p, max_new_tokens=40))
    print("-" * 50)